#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300指数择时模型 (Index Timing Model)

专为大盘仓位控制设计：预测沪深300未来 N 个交易日的涨跌方向，
输出 pred_prob → 仓位档位映射（满仓 / 半仓 / 空仓）。

可与截面选股模型组合使用：
  择时模型 → 决定总仓位（槽位数量：0 / 10 / 20）
  截面模型 → 决定选哪些股票

特征（~29个，均基于 T 日收盘可知信息）：
  指数技术面 (15): 多尺度收益率/波动率/MA偏离/RSI/布林带位置/成交量比
  市场宽度   (5):  等权市场收益/截面波动率/上涨比例/MA20上方比例/涨跌比
  宏观北向   (5):  北向净流入1/5/20日、南向、北向加速度
  宏观月度   (3):  制造业PMI（相对50的偏差）/ M2同比 / SHIBOR 3m
  时间编码   (4):  月份/星期 sin-cos 变换

标签：
  (close_{T+N} / close_T - 1) > 0  → 1（指数上涨，做多信号有效）
                                    → 0（指数下跌或持平，持现金）

仓位映射（基于 pred_prob）：
  ≥ threshold_full → 满仓（slots_full 槽，默认20）
  ≥ threshold_half → 半仓（slots_half 槽，默认10）
  < threshold_half → 空仓（0槽）

使用方式:
  python index_timing_model.py
  python index_timing_model.py --rebal_freq 10 --threshold_full 0.60
  python index_timing_model.py --no_wfo          # 跳过WFO，快速验证
"""

import os
import warnings
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix
)
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# 1. 配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # 数据源
    db_path: str = "data/quant.duckdb"
    index_code: str = "000300.SH"
    data_start: str = "20160101"      # 特征起始（需留足 MA250 预热）

    # 时间切分
    train_end: str   = "20221231"     # 训练集结束（含2022熊市，提高泛化）
    val_cutoff: str  = "20220101"     # 验证集起点（训练集内部早停用）
    val_end: str     = "20221231"     # 验证集结束
    test_start: str  = "20230201"     # 测试集起点（含 20 日隔离期）
    end_date: str    = "20260311"
    pred_suffix: str = ""             # 预测文件后缀（prod 模式下设为 "_prod"）

    # 标签
    rebal_freq: int = 15              # 前向持有期（交易日），与选股模型对齐

    # XGBoost（防过拟合配置：时序样本少，精选16个特征后适度复杂度）
    n_estimators: int         = 150   # 适中树数（精选特征 + 强正则后不易过拟合）
    max_depth: int            = 2     # 最浅，只捕获2阶交互
    learning_rate: float      = 0.05
    subsample: float          = 0.80
    colsample_bytree: float   = 0.60  # 每棵树用60%特征（≈9-10个）
    min_child_weight: int     = 30    # 叶节点至少30个样本才分裂
    reg_lambda: float         = 15.0  # 强L2正则
    reg_alpha: float          = 1.5   # 强L1正则
    # 注: 不使用 early_stopping，固定 n_estimators + 强正则化替代

    # 仓位档位（保守阈值：让模型默认落在半仓，只在高置信度时才满仓/空仓）
    threshold_full: float = 0.60     # ≥ 此值 → 满仓（需高置信度看涨）
    threshold_half: float = 0.45     # ≥ 此值 → 半仓；< 此值 → 空仓（低置信或看跌）
    slots_full: int  = 20
    slots_half: int  = 10
    slots_empty: int = 0

    # 滚动百分位仓位映射（解决模型概率跨时期聚集问题）
    # 原理：将当期概率与近期历史比较，用相对排名决定仓位，消除绝对阈值的跨期失效
    use_rolling_pct: bool  = False   # True → 用相对百分位; False → 用绝对阈值
    pct_window:      int   = 30      # 滚动窗口大小（调仓期数）
    pct_warmup:      int   = 10      # 预热期：不足此期数时仍用绝对阈值
    pct_full:        float = 0.70    # 百分位 ≥ 此值 → 满仓（近期最看涨前30%）
    pct_half:        float = 0.35    # 百分位 ≥ 此值 → 半仓；< 此值 → 空仓

    # 成本（评估择时收益用）
    commission: float  = 0.0003      # 双边佣金
    stamp_tax: float   = 0.001       # 印花税（仅卖出）
    slippage: float    = 0.001       # 双边滑点

    # 实验选项
    # label_type: 预测目标
    #   'direction'  → sign(close_{T+N} - close_T)（当前，接近随机游走）
    #   'ma60_state' → I(close_{T+N} > MA60_{T+N})（MA60状态持续性，理论上更可预测）
    #   'meta_label' → 元标签（Lopez de Prado）：仅对 MA 规则活跃期（close≥MA20）标注
    #                  标签 = 1 若 MA 信号盈利（fwd_ret > 0），否则 0
    #                  ML 目标：预测"MA 信号是否可信"，而非预测市场方向
    #                  仓位逻辑：MA 确定上限，ML 决定是否执行
    label_type: str = 'direction'

    # MA硬覆盖模式
    #   'hard3'    → 原始三档强制（close<MA20→空, MA20-60→半, MA60+→ML）【默认】
    #   'ma20only' → 仅 MA20 硬保护，MA20-MA60区域让ML自由（适合ma60_state前瞻标签）
    #                close<MA20→空仓; close≥MA20→ML自由（取消MA60半仓限制）
    #   'soft'    → 仅当ML极低置信时才覆盖（适用于ma60_state标签：ML直接预测MA60位置）
    #               close<MA20 且 prob<0.40 → 空仓；close<MA60 且 prob<pct_half → 半仓
    #   'none'    → 关闭MA覆盖，完全由ML决定（诊断用）
    ma_override: str = 'hard3'

    # 输出
    output_dir: str = "output"


# 精选特征子集（16个）：去掉短期噪声特征，只保留稳健的中长期信号
# 研究表明：短期收益率（1/3/5/10日）和短期波动（5日）对指数方向预测噪声过大
SELECTED_FEATURES = [
    # 中期动量（20d/60d/120d）
    'ret_20d', 'ret_60d', 'ret_120d',
    # 波动率状态（20d/60d年化）
    'vol_20d', 'vol_60d',
    # MA偏离：价格相对均线位置（趋势强弱）
    'close_vs_ma20', 'close_vs_ma60', 'close_vs_ma120', 'close_vs_ma250',
    # 市场宽度
    'breadth_pct_ma20',
    # 宏观信号
    'm2_yoy', 'pmi_vs_50', 'shibor_3m',
    # 北向资金（20日累积）
    'north_20d',
    # 季节性编码
    'month_sin', 'month_cos',
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. 特征工程
# ══════════════════════════════════════════════════════════════════════════════

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI（指数移动平均法）"""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def load_index_features(conn, cfg: Config) -> pd.DataFrame:
    """
    沪深300指数技术特征（15个）：
    收益率 × 7 / 波动率 × 3 / MA偏离 × 4 / RSI / 布林带 / 成交量比
    """
    print("  加载指数日线...")
    df = conn.execute(f"""
        SELECT trade_date, close, pct_chg, vol, amount
        FROM index_daily
        WHERE ts_code = '{cfg.index_code}'
          AND trade_date >= '{cfg.data_start}'
          AND trade_date <= '{cfg.end_date}'
        ORDER BY trade_date
    """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.set_index('trade_date')

    close = df['close']
    ret   = df['pct_chg'] / 100.0
    vol   = df['vol']

    f = pd.DataFrame(index=df.index)

    # 多尺度收益率
    for d in [1, 3, 5, 10, 20, 60, 120]:
        f[f'ret_{d}d'] = close / close.shift(d) - 1

    # 年化波动率
    for d in [5, 20, 60]:
        f[f'vol_{d}d'] = ret.rolling(d).std() * np.sqrt(252)

    # MA 偏离（收盘/MA - 1）
    for d in [20, 60, 120, 250]:
        f[f'close_vs_ma{d}'] = close / close.rolling(d).mean() - 1

    # RSI(14)
    f['rsi_14'] = _compute_rsi(close, 14)

    # 布林带位置 (20日, 2σ)：0 = 下轨，1 = 上轨
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    f['bb_pos'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # 成交量比（当日 / 20日均量）
    f['vol_ratio_20d'] = vol / vol.rolling(20).mean()

    # 季节性编码（月份 sin/cos）
    idx_dt = pd.to_datetime(df.index, format='%Y%m%d')
    f['month_sin'] = np.sin(2 * np.pi * idx_dt.month / 12)
    f['month_cos'] = np.cos(2 * np.pi * idx_dt.month / 12)

    print(f"    指数特征: {len(f)} 天 × {len(f.columns)} 列")
    return f


def load_breadth_features(conn, cfg: Config) -> pd.DataFrame:
    """
    市场宽度特征（5个）：
    等权市场收益 / 截面波动率 / 上涨比例 / MA20上方比例 / 涨跌家数比
    （SQL聚合，不过滤8/4开头北交所等，保持全市场）
    """
    print("  计算市场宽度（SQL聚合，约30秒）...")
    df = conn.execute(f"""
        WITH dp AS (
            SELECT ts_code, trade_date, close, pct_chg
            FROM daily_price
            WHERE trade_date >= '{cfg.data_start}'
              AND trade_date <= '{cfg.end_date}'
        ),
        with_ma AS (
            SELECT
                ts_code, trade_date, close, pct_chg,
                AVG(close) OVER (
                    PARTITION BY ts_code ORDER BY trade_date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) AS ma20
            FROM dp
        )
        SELECT
            trade_date,
            AVG(pct_chg / 100.0)                                         AS mkt_ret_ew,
            STDDEV(pct_chg / 100.0) * SQRT(252)                          AS mkt_cross_vol,
            SUM(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END)
                / NULLIF(COUNT(*), 0)                                     AS advance_ratio,
            SUM(CASE WHEN close > ma20 THEN 1.0 ELSE 0.0 END)
                / NULLIF(COUNT(*), 0)                                     AS breadth_pct_ma20,
            SUM(CASE WHEN pct_chg > 0 THEN 1.0 ELSE 0.0 END)
                / NULLIF(SUM(CASE WHEN pct_chg < 0 THEN 1.0 ELSE 0.0 END), 0)
                                                                          AS adv_dec_ratio
        FROM with_ma
        GROUP BY trade_date
        ORDER BY trade_date
    """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.set_index('trade_date')

    print(f"    市场宽度: {len(df)} 天")
    return df


def load_macro_features(conn, cfg: Config, trading_dates: List[str]) -> pd.DataFrame:
    """
    宏观 + 北向资金特征（8个）：
    北向净流入 1/5/20日 + 南向 + 北向加速度 + PMI偏差 + M2同比 + SHIBOR 3m

    月度数据按交易日历前向填充（PMI/M2 对应发布月后1个月，保守处理）。
    """
    print("  加载宏观/北向特征...")

    # ── 北向资金（日度）──────────────────────────────────────────────────
    hsgt = conn.execute(f"""
        SELECT trade_date, north_money, south_money
        FROM moneyflow_hsgt
        WHERE trade_date >= '{cfg.data_start}'
          AND trade_date <= '{cfg.end_date}'
        ORDER BY trade_date
    """).fetchdf()
    hsgt['trade_date'] = hsgt['trade_date'].astype(str)
    hsgt = hsgt.set_index('trade_date')
    hsgt['north_1d']   = hsgt['north_money']
    hsgt['north_5d']   = hsgt['north_money'].rolling(5).sum()
    hsgt['north_20d']  = hsgt['north_money'].rolling(20).sum()
    # 北向加速度：5日净流 - 20日净流/4（短期趋势 vs 中期基线）
    hsgt['north_accel'] = hsgt['north_5d'] - hsgt['north_20d'] / 4
    hsgt = hsgt[['north_1d', 'north_5d', 'north_20d', 'south_money', 'north_accel']]

    # ── PMI（月度，YYYYMM）──────────────────────────────────────────────
    pmi_raw = conn.execute("SELECT month, pmi FROM cn_pmi ORDER BY month").fetchdf()
    # 保守处理：PMI 发布月 M → 从下月第 1 个交易日起可用
    # 将 YYYYMM → 下个月的 YYYYMM01 字符串，再 ffill 到日历
    def next_month_str(yyyymm: str) -> str:
        y, m = int(yyyymm[:4]), int(yyyymm[4:6])
        m += 1
        if m > 12:
            m, y = 1, y + 1
        return f"{y}{m:02d}01"

    pmi_raw['avail_from'] = pmi_raw['month'].astype(str).apply(next_month_str)
    pmi_raw['pmi_vs_50']  = pmi_raw['pmi'] - 50.0
    pmi_raw = pmi_raw.set_index('avail_from')[['pmi', 'pmi_vs_50']]

    # ── M2（月度）───────────────────────────────────────────────────────
    m2_raw = conn.execute("SELECT month, m2_yoy FROM cn_m ORDER BY month").fetchdf()
    m2_raw['avail_from'] = m2_raw['month'].astype(str).apply(next_month_str)
    m2_raw = m2_raw.set_index('avail_from')[['m2_yoy']]

    # ── SHIBOR（日度，date 列）──────────────────────────────────────────
    shibor = conn.execute(f"""
        SELECT date AS trade_date, m3 AS shibor_3m
        FROM shibor
        WHERE date >= '{cfg.data_start}' AND date <= '{cfg.end_date}'
        ORDER BY date
    """).fetchdf()
    shibor['trade_date'] = shibor['trade_date'].astype(str)
    shibor = shibor.set_index('trade_date')[['shibor_3m']]

    # ── 拼到交易日历 ────────────────────────────────────────────────────
    macro = pd.DataFrame(index=trading_dates)

    # 月度数据：以 YYYYMM01 为 key，将 macro 行索引取前 6 位作月份 key
    # 构建月度映射 dict，再 map → ffill
    pmi_pmi_dict   = pmi_raw['pmi'].to_dict()
    pmi_vs50_dict  = pmi_raw['pmi_vs_50'].to_dict()
    m2_dict        = m2_raw['m2_yoy'].to_dict()

    # 将交易日转为 YYYYMM01（方便 lookup）
    def date_to_month01(d: str) -> str:
        return d[:6] + '01'

    macro['_month01'] = [date_to_month01(d) for d in macro.index]
    macro['pmi']       = macro['_month01'].map(pmi_pmi_dict)
    macro['pmi_vs_50'] = macro['_month01'].map(pmi_vs50_dict)
    macro['m2_yoy']    = macro['_month01'].map(m2_dict)
    macro = macro.drop(columns=['_month01'])
    macro[['pmi', 'pmi_vs_50', 'm2_yoy']] = macro[['pmi', 'pmi_vs_50', 'm2_yoy']].ffill()

    # 日度数据：左连接
    macro = macro.join(shibor, how='left')
    macro['shibor_3m'] = macro['shibor_3m'].ffill()
    macro = macro.join(hsgt, how='left')
    macro[hsgt.columns] = macro[hsgt.columns].ffill()

    print(f"    宏观特征: {len(macro)} 天 × {len(macro.columns)} 列")
    return macro


def build_panel(cfg: Config) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    合并所有特征，计算标签，返回 (panel, feat_cols, idx_close)。
    """
    conn = duckdb.connect(cfg.db_path, read_only=True)

    idx_feat    = load_index_features(conn, cfg)
    breadth     = load_breadth_features(conn, cfg)

    trading_dates = idx_feat.index.tolist()
    macro       = load_macro_features(conn, cfg, trading_dates)

    # 指数收盘价（用于标签和评估）
    idx_close_df = conn.execute(f"""
        SELECT trade_date, close
        FROM index_daily
        WHERE ts_code = '{cfg.index_code}'
          AND trade_date >= '{cfg.data_start}'
          AND trade_date <= '{cfg.end_date}'
        ORDER BY trade_date
    """).fetchdf()
    idx_close_df['trade_date'] = idx_close_df['trade_date'].astype(str)
    idx_close = idx_close_df.set_index('trade_date')['close']
    conn.close()

    # 合并所有特征
    panel = idx_feat.copy()
    panel = panel.join(breadth, how='left')
    panel = panel.join(macro,   how='left')

    # 标签构造（支持两种预测目标）
    fwd_ret  = idx_close.shift(-cfg.rebal_freq) / idx_close - 1
    panel['fwd_ret'] = fwd_ret

    if cfg.label_type == 'ma60_state':
        # ── MA60 状态标签 ─────────────────────────────────────────────
        # 预测目标：rebal_freq 日后 close 是否在 MA60 之上
        # 优势：MA60 状态高度持续（自相关），close_vs_ma60 与标签关系物理稳定
        future_close = idx_close.shift(-cfg.rebal_freq)
        future_ma60  = idx_close.rolling(60).mean().shift(-cfg.rebal_freq)
        label_raw    = (future_close > future_ma60).astype(float)
        label_raw[future_close.isna() | future_ma60.isna()] = np.nan
        panel['label'] = label_raw
        label_desc = 'MA60状态（close_{T+N} > MA60_{T+N}）'

    elif cfg.label_type == 'meta_label':
        # ── 元标签（Lopez de Prado Meta-Labeling）──────────────────────
        # 核心思想：将策略决策拆为两层
        #   主模型（primary）：MA规则确定方向 → close≥MA60: 满仓信号; MA20-MA60: 半仓信号
        #   元标签器（secondary ML）：预测"主模型信号是否会盈利"
        #
        # 为什么有效：
        #   1. 训练集更纯净：只训练 MA 规则活跃期（close≥MA20），剔除了"趋势不明"的噪声期
        #   2. 标签正样本率更高：MA 活跃期天然具有上涨倾向（约60-65%），vs 总体53%
        #   3. ML 任务更聚焦："区分盈利/亏损的MA信号"比"预测市场方向"更有学习空间
        #   4. MA 规则提供结构性保护：close<MA20 永远不交易，ML无需学习熊市规避
        #
        # 标签构造规则：
        #   close_T < MA20_T  → label = NaN（主模型未发信号，不纳入训练集）
        #   close_T ≥ MA20_T  → label = 1 若 fwd_ret > 0，否则 0
        #                       （主模型给了信号，这个信号是否盈利？）
        ma20 = idx_close.rolling(20).mean()
        primary_active = idx_close >= ma20   # MA 规则在 T 日发出了信号

        label_raw = pd.Series(np.nan, index=panel.index)
        profit_mask = primary_active & (fwd_ret > 0)
        loss_mask   = primary_active & (fwd_ret <= 0) & fwd_ret.notna()
        label_raw[profit_mask] = 1.0
        label_raw[loss_mask]   = 0.0
        panel['label']    = label_raw
        # 存储 primary_active 状态，供 simulate_timing 使用
        panel['_meta_primary_active'] = primary_active.astype(float)
        label_desc = (f'元标签（MA规则活跃期收益方向，'
                      f'活跃率={primary_active.mean():.1%}）')

    else:
        # ── 方向标签（原始，接近随机游走）────────────────────────────
        panel['label'] = (fwd_ret > 0).astype(float)
        panel.loc[fwd_ret.isna(), 'label'] = np.nan
        label_desc = f'{cfg.rebal_freq}日前向收益方向（>0→1）'

    # 丢弃最后 rebal_freq 行（标签未知）
    panel = panel.iloc[: -cfg.rebal_freq]

    # 使用精选特征子集（稳健的中长期信号）
    avail = [c for c in SELECTED_FEATURES if c in panel.columns]
    missing = [c for c in SELECTED_FEATURES if c not in panel.columns]
    if missing:
        print(f"  警告：以下特征不存在，已跳过: {missing}")
    feat_cols = avail

    pos_rate = panel['label'].dropna().mean()
    print(f"\n  面板汇总: {len(panel)} 行 × {len(feat_cols)} 特征（精选）")
    print(f"  标签类型: {label_desc}")
    print(f"  正样本率: {pos_rate:.1%}")
    print(f"  特征列: {feat_cols}")
    return panel, feat_cols, idx_close


# ══════════════════════════════════════════════════════════════════════════════
# 3. 训练
# ══════════════════════════════════════════════════════════════════════════════

def _purged_cutoff(val_start: str, embargo: int, all_dates: List[str]) -> str:
    """训练集截止日 = val_start 向前移 embargo 个交易日"""
    try:
        idx = all_dates.index(val_start)
    except ValueError:
        idx = next((i for i, d in enumerate(all_dates) if d >= val_start), len(all_dates))
    cutoff_idx = max(0, idx - embargo)
    return all_dates[cutoff_idx]


def train_model(
    panel: pd.DataFrame,
    feat_cols: List[str],
    cfg: Config,
    val_start: str,
    val_end: str,
    label: str = "主模型",
) -> xgb.XGBClassifier:
    """
    训练 XGBoost 二分类器，使用 Purged 分割避免标签泄露。

    训练集：data_start ~ purged_cutoff（val_start 前 rebal_freq 日）
    验证集：val_start ~ val_end（用于 Early Stopping）
    """
    all_dates = sorted(panel.index.dropna().tolist())
    purged_end = _purged_cutoff(val_start, cfg.rebal_freq, all_dates)

    train_mask = (panel.index <= purged_end)
    val_mask   = (panel.index >= val_start) & (panel.index <= val_end)

    # 只用特征全部可用的行（去掉MA250等长窗口特征的预热期）
    train = panel[train_mask].dropna(subset=feat_cols + ['label'])
    val   = panel[val_mask].dropna(subset=feat_cols + ['label'])

    if len(train) == 0 or len(val) == 0:
        raise ValueError(f"[{label}] 训练集({len(train)})或验证集({len(val)})为空")

    X_tr,  y_tr  = train[feat_cols], train['label']
    X_val, y_val = val[feat_cols],   val['label']

    # 不使用 early stopping：时序数据验证集太小，早停信号噪声过大
    # 改用强正则化 + 固定轮次确保模型不过拟合
    mdl = xgb.XGBClassifier(
        n_estimators     = cfg.n_estimators,
        max_depth        = cfg.max_depth,
        learning_rate    = cfg.learning_rate,
        subsample        = cfg.subsample,
        colsample_bytree = cfg.colsample_bytree,
        min_child_weight = cfg.min_child_weight,
        reg_lambda       = cfg.reg_lambda,
        reg_alpha        = cfg.reg_alpha,
        objective        = 'binary:logistic',
        eval_metric      = 'logloss',
        tree_method      = 'hist',
        random_state     = 42,
        n_jobs           = -1,
    )
    mdl.fit(X_tr, y_tr, verbose=False)

    # 后扫描最优阈值（Youden J 统计量最大化）
    val_prob = mdl.predict_proba(X_val)[:, 1]
    best_thresh, best_j = 0.50, -1.0
    for t in np.arange(0.30, 0.71, 0.02):
        pred = (val_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        j   = tpr + tnr - 1
        if j > best_j:
            best_j, best_thresh = j, t
    mdl._best_threshold = best_thresh

    tr_auc  = roc_auc_score(y_tr,  mdl.predict_proba(X_tr)[:, 1])
    val_auc = roc_auc_score(y_val, val_prob)
    print(f"  [{label}] 训练={len(train)}行  验证={len(val)}行  "
          f"最优阈值={best_thresh:.2f}  "
          f"train_AUC={tr_auc:.3f}  val_AUC={val_auc:.3f}  "
          f"n_trees={cfg.n_estimators}")
    return mdl


# ══════════════════════════════════════════════════════════════════════════════
# 4. 评估 & 组合模拟
# ══════════════════════════════════════════════════════════════════════════════

def prob_to_slots(prob: float, cfg: Config) -> int:
    """pred_prob → 槽位数量"""
    if prob >= cfg.threshold_full:
        return cfg.slots_full
    elif prob >= cfg.threshold_half:
        return cfg.slots_half
    return cfg.slots_empty


def meta_to_slots(prob: float, ma_state: str, cfg: Config) -> int:
    """
    元标签仓位映射：MA 规则确定方向上限，ML 决定是否执行。

    ma_state:
      'flat'  → 主模型未发信号（close < MA20），永远空仓
      'half'  → 主模型建议半仓（MA20 ≤ close < MA60）
                ML yes(≥th_half) → 半仓；ML no → 空仓
      'long'  → 主模型建议满仓（close ≥ MA60）
                ML yes(≥th_full) → 满仓；ML maybe(≥th_half) → 半仓；ML no → 空仓
    """
    if ma_state == 'flat':
        return cfg.slots_empty
    elif ma_state == 'half':
        return cfg.slots_half if prob >= cfg.threshold_half else cfg.slots_empty
    else:  # 'long'
        if prob >= cfg.threshold_full:
            return cfg.slots_full
        elif prob >= cfg.threshold_half:
            return cfg.slots_half
        return cfg.slots_empty


def simulate_timing(
    prob_series: pd.Series,
    idx_close: pd.Series,
    cfg: Config,
    date_start: str,
    panel: Optional[pd.DataFrame] = None,   # 用于 MA 硬覆盖
    date_end: Optional[str] = None,
    label: str = "测试集",
) -> dict:
    """
    模拟择时策略：每 rebal_freq 日调仓，按 prob → slots 决定持仓比例。

    仓位决策逻辑（混合 ML + MA 规则）：
      Step 1 — ML 信号：
        - 绝对阈值模式：prob ≥ threshold_full → 满仓；≥ threshold_half → 半仓；else → 空仓
        - 滚动百分位模式（use_rolling_pct=True）：用近期历史概率的相对排名映射仓位，
          消除因跨时期概率分布漂移导致的"永久半仓"问题
      Step 2 — 3档 MA 硬覆盖（安全网，不受 ML 影响）：
        close < MA20 → 强制空仓（深熊/崩盘保护）
        MA20 ≤ close < MA60 → 强制半仓（震荡/调整区）
        close ≥ MA60 → ML 自由决定（上升趋势）

    收益计算：port_ret = weight × index_ret_period - cost
    weight = slots / slots_full → {0, 0.5, 1.0}
    """
    # 筛选日期范围
    mask = prob_series.index >= date_start
    if date_end:
        mask &= (prob_series.index <= date_end)
    probs = prob_series[mask].sort_index()

    # 按 rebal_freq 间隔取调仓日
    all_prob_dates = probs.index.tolist()
    rebal_dates    = all_prob_dates[::cfg.rebal_freq]

    port_rets, bh_rets, slot_hist = [], [], []
    prob_history: List[float] = []   # 滚动百分位窗口（调仓期顺序）

    for i in range(len(rebal_dates) - 1):
        d      = rebal_dates[i]
        d_next = rebal_dates[i + 1]

        if d not in probs.index:
            continue
        p0 = idx_close.get(d,      np.nan)
        p1 = idx_close.get(d_next, np.nan)
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
            continue

        idx_ret = p1 / p0 - 1
        prob    = probs[d]

        # ── Step 1 & 2：仓位决策（两条路径）───────────────────────────
        if cfg.label_type == 'meta_label' and panel is not None and d in panel.index:
            # ── 元标签路径：MA 规则确定上限，ML 决定是否执行 ───────────
            # MA 状态由当期的 close_vs_ma20 / close_vs_ma60 决定
            ma60_dev = panel.at[d, 'close_vs_ma60'] if 'close_vs_ma60' in panel.columns else np.nan
            ma20_dev = panel.at[d, 'close_vs_ma20'] if 'close_vs_ma20' in panel.columns else np.nan
            if not np.isfinite(ma60_dev): ma60_dev = 0.0
            if not np.isfinite(ma20_dev): ma20_dev = 0.0
            if ma20_dev < 0:
                ma_state = 'flat'
            elif ma60_dev < 0:
                ma_state = 'half'
            else:
                ma_state = 'long'
            slots = meta_to_slots(prob, ma_state, cfg)
            prob_history.append(prob)
        else:
            # ── 标准路径：ML → slots，然后 MA 覆盖 ──────────────────────
            if cfg.use_rolling_pct and len(prob_history) >= cfg.pct_warmup:
                window   = prob_history[-cfg.pct_window:]
                pct_rank = float(np.mean(np.array(window) <= prob))
                if pct_rank >= cfg.pct_full:
                    slots = cfg.slots_full
                elif pct_rank >= cfg.pct_half:
                    slots = cfg.slots_half
                else:
                    slots = cfg.slots_empty
            else:
                slots = prob_to_slots(prob, cfg)

            prob_history.append(prob)

            # MA 覆盖
            if panel is not None and d in panel.index and cfg.ma_override != 'none':
                ma60_dev = panel.at[d, 'close_vs_ma60'] if 'close_vs_ma60' in panel.columns else np.nan
                ma20_dev = panel.at[d, 'close_vs_ma20'] if 'close_vs_ma20' in panel.columns else np.nan
                if not np.isfinite(ma60_dev): ma60_dev = 0.0
                if not np.isfinite(ma20_dev): ma20_dev = 0.0
                if cfg.ma_override == 'hard3':
                    if ma20_dev < 0:
                        slots = cfg.slots_empty
                    elif ma60_dev < 0:
                        slots = min(slots, cfg.slots_half)
                elif cfg.ma_override == 'ma20only':
                    if ma20_dev < 0:
                        slots = cfg.slots_empty
                elif cfg.ma_override == 'soft':
                    if ma20_dev < -0.03 and prob < 0.40:
                        slots = cfg.slots_empty
                    elif ma60_dev < 0 and prob < cfg.threshold_half:
                        slots = min(slots, cfg.slots_half)

        weight  = slots / cfg.slots_full   # 0 / 0.5 / 1.0

        # 交易成本
        prev_weight = slot_hist[-1] / cfg.slots_full if slot_hist else 0.0
        delta = abs(weight - prev_weight)
        cost  = 0.0
        if delta > 0:
            cost += delta * (cfg.commission + cfg.slippage)
            if weight < prev_weight:             # 有卖出
                cost += delta * cfg.stamp_tax

        port_rets.append(weight * idx_ret - cost)
        bh_rets.append(idx_ret)
        slot_hist.append(slots)

    if len(port_rets) == 0:
        return {'label': label}

    pr  = np.array(port_rets)
    bhr = np.array(bh_rets)
    ann = 252 / cfg.rebal_freq

    def _stats(rets):
        r_ann = np.mean(rets) * ann
        v_ann = np.std(rets) * np.sqrt(ann)
        sharpe = r_ann / v_ann if v_ann > 0 else 0.0
        cum = np.cumprod(1 + rets)
        dd  = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
        maxdd = float(dd.min())
        calmar = r_ann / abs(maxdd) if maxdd < 0 else np.nan
        total = float(cum[-1]) - 1
        return dict(ann=r_ann, vol=v_ann, sharpe=sharpe, maxdd=maxdd, calmar=calmar, total=total)

    ps  = _stats(pr)
    bhs = _stats(bhr)

    slots_arr = np.array(slot_hist)
    full_pct  = (slots_arr == cfg.slots_full).mean()
    half_pct  = (slots_arr == cfg.slots_half).mean()
    empty_pct = (slots_arr == cfg.slots_empty).mean()

    print(f"\n  ── {label} 仓位分配 ──")
    print(f"    满仓 {full_pct:.1%} | 半仓 {half_pct:.1%} | 空仓 {empty_pct:.1%}")
    print(f"    择时: 年化 {ps['ann']:.2%}  Sharpe {ps['sharpe']:.3f}  "
          f"MaxDD {ps['maxdd']:.2%}  总收益 {ps['total']:.2%}")
    print(f"    买持: 年化 {bhs['ann']:.2%}  Sharpe {bhs['sharpe']:.3f}  "
          f"MaxDD {bhs['maxdd']:.2%}  总收益 {bhs['total']:.2%}")

    return dict(
        label=label,
        port_ann=ps['ann'], port_sharpe=ps['sharpe'],
        port_maxdd=ps['maxdd'], calmar=ps['calmar'], port_total=ps['total'],
        bh_ann=bhs['ann'], bh_sharpe=bhs['sharpe'], bh_maxdd=bhs['maxdd'],
        full_pct=full_pct, half_pct=half_pct, empty_pct=empty_pct,
        port_rets=pr, bh_rets=bhr, slot_counts=slot_hist,
        rebal_dates=rebal_dates[:len(pr)],
        n_periods=len(pr),
    )


def simulate_ma_baseline(
    panel: pd.DataFrame,
    idx_close: pd.Series,
    cfg: Config,
    date_start: str,
    date_end: Optional[str] = None,
    label: str = "MA基准",
) -> dict:
    """
    纯 MA 规则择时基准（无 ML）：
      close < MA20  → 空仓（0槽）
      MA20 ≤ close < MA60 → 半仓（10槽）
      close ≥ MA60 → 满仓（20槽）

    用于对比 ML 模型是否比简单 MA 规则有额外价值。
    """
    mask = panel.index >= date_start
    if date_end:
        mask &= (panel.index <= date_end)
    panel_sub = panel[mask]
    all_dates  = panel_sub.index.tolist()
    rebal_dates = all_dates[::cfg.rebal_freq]

    port_rets, bh_rets, slot_hist = [], [], []

    for i in range(len(rebal_dates) - 1):
        d      = rebal_dates[i]
        d_next = rebal_dates[i + 1]
        if d not in panel_sub.index:
            continue
        p0 = idx_close.get(d,      np.nan)
        p1 = idx_close.get(d_next, np.nan)
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
            continue
        idx_ret = p1 / p0 - 1

        # 纯 MA 3档规则
        ma20_dev = panel_sub.at[d, 'close_vs_ma20'] if 'close_vs_ma20' in panel_sub.columns else 0.0
        ma60_dev = panel_sub.at[d, 'close_vs_ma60'] if 'close_vs_ma60' in panel_sub.columns else 0.0
        if not np.isfinite(ma20_dev): ma20_dev = 0.0
        if not np.isfinite(ma60_dev): ma60_dev = 0.0

        if ma20_dev < 0:
            slots = cfg.slots_empty
        elif ma60_dev < 0:
            slots = cfg.slots_half
        else:
            slots = cfg.slots_full

        weight      = slots / cfg.slots_full
        prev_weight = slot_hist[-1] / cfg.slots_full if slot_hist else 0.0
        delta = abs(weight - prev_weight)
        cost  = 0.0
        if delta > 0:
            cost += delta * (cfg.commission + cfg.slippage)
            if weight < prev_weight:
                cost += delta * cfg.stamp_tax

        port_rets.append(weight * idx_ret - cost)
        bh_rets.append(idx_ret)
        slot_hist.append(slots)

    if len(port_rets) == 0:
        return {'label': label}

    pr  = np.array(port_rets)
    bhr = np.array(bh_rets)
    ann = 252 / cfg.rebal_freq

    def _stats(rets):
        r_ann  = np.mean(rets) * ann
        v_ann  = np.std(rets) * np.sqrt(ann)
        sharpe = r_ann / v_ann if v_ann > 0 else 0.0
        cum    = np.cumprod(1 + rets)
        dd     = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
        maxdd  = float(dd.min())
        calmar = r_ann / abs(maxdd) if maxdd < 0 else np.nan
        total  = float(cum[-1]) - 1
        return dict(ann=r_ann, vol=v_ann, sharpe=sharpe, maxdd=maxdd, calmar=calmar, total=total)

    ps  = _stats(pr)
    bhs = _stats(bhr)

    slots_arr = np.array(slot_hist)
    full_pct  = (slots_arr == cfg.slots_full).mean()
    half_pct  = (slots_arr == cfg.slots_half).mean()
    empty_pct = (slots_arr == cfg.slots_empty).mean()

    print(f"\n  ── {label} 仓位分配 ──")
    print(f"    满仓 {full_pct:.1%} | 半仓 {half_pct:.1%} | 空仓 {empty_pct:.1%}")
    print(f"    MA基准: 年化 {ps['ann']:.2%}  Sharpe {ps['sharpe']:.3f}  "
          f"MaxDD {ps['maxdd']:.2%}  总收益 {ps['total']:.2%}")
    print(f"    买持:   年化 {bhs['ann']:.2%}  Sharpe {bhs['sharpe']:.3f}  "
          f"MaxDD {bhs['maxdd']:.2%}  总收益 {bhs['total']:.2%}")

    return dict(
        label=label,
        port_ann=ps['ann'], port_sharpe=ps['sharpe'],
        port_maxdd=ps['maxdd'], calmar=ps['calmar'], port_total=ps['total'],
        bh_ann=bhs['ann'], bh_sharpe=bhs['sharpe'], bh_maxdd=bhs['maxdd'],
        full_pct=full_pct, half_pct=half_pct, empty_pct=empty_pct,
        port_rets=pr, bh_rets=bhr, slot_counts=slot_hist,
        rebal_dates=rebal_dates[:len(pr)],
        n_periods=len(pr),
    )


def evaluate_classification(
    panel: pd.DataFrame,
    prob_series: pd.Series,
    date_start: str,
    date_end: Optional[str] = None,
    label: str = "测试集",
) -> dict:
    """AUC / Brier / Accuracy / 混淆矩阵"""
    mask = (panel.index >= date_start)
    if date_end:
        mask &= (panel.index <= date_end)
    sub   = panel[mask].dropna(subset=['label'])
    common = sub.index.intersection(prob_series.index)
    y_true = sub.loc[common, 'label'].values
    y_prob = prob_series.loc[common].values

    if len(np.unique(y_true)) < 2:
        return {}

    auc   = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    acc   = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    cm    = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))

    print(f"\n  ── {label} 分类指标 ──")
    print(f"    AUC={auc:.3f}  Brier={brier:.3f}  Acc={acc:.1%}")
    cm_str = f"    [[TN={cm[0,0]} FP={cm[0,1]}]\n     [FN={cm[1,0]} TP={cm[1,1]}]]"
    print(cm_str)

    return dict(auc=auc, brier=brier, accuracy=acc, confusion_matrix=cm)


def walk_forward_evaluation(
    panel: pd.DataFrame,
    feat_cols: List[str],
    idx_close: pd.Series,
    cfg: Config,
    wfo_years: Optional[List[int]] = None,
) -> List[dict]:
    """
    滚动样本外验证（WFO）：
    每次以 [data_start, year] 为训练+验证，[year+1] 为测试。
    """
    if wfo_years is None:
        wfo_years = [2021, 2022, 2023, 2024]   # 每年 = val 年，测试 = val+1

    all_dates = sorted(panel.index.tolist())
    results   = []

    for year in wfo_years:
        val_start = f"{year}0101"
        val_end   = f"{year}1231"
        test_start = f"{year + 1}0101"
        test_end   = f"{year + 1}1231"

        # 找第一个可用日期
        val_start_avail = next((d for d in all_dates if d >= val_start), None)
        test_start_avail = next((d for d in all_dates if d >= test_start), None)

        if test_start_avail is None or test_start_avail > all_dates[-1]:
            print(f"  [WFO {year+1}] 无测试数据，跳过")
            continue

        print(f"\n[WFO {year+1}] 训练至 {year-1}  验证={year}  测试={year+1}")
        try:
            mdl = train_model(
                panel, feat_cols, cfg,
                val_start=val_start_avail,
                val_end=val_end,
                label=f"WFO {year+1}",
            )
        except Exception as e:
            print(f"  训练失败: {e}")
            continue

        test_panel = panel[
            (panel.index >= test_start_avail) & (panel.index <= test_end)
        ].dropna(subset=feat_cols[:3])   # 至少前几个特征非空

        if len(test_panel) < cfg.rebal_freq * 2:
            print(f"  [WFO {year+1}] 测试集太短（{len(test_panel)}行），跳过")
            continue

        test_prob = pd.Series(
            mdl.predict_proba(test_panel[feat_cols])[:, 1],
            index=test_panel.index,
        )
        # WFO 使用固定的 cfg 阈值（不受 val-scan 干扰），保持一致的仓位逻辑
        sim = simulate_timing(
            test_prob, idx_close, cfg,
            date_start=test_start_avail, panel=test_panel,
            date_end=test_end,
            label=f"WFO {year+1}",
        )
        cls = evaluate_classification(
            test_panel, test_prob,
            date_start=test_start_avail, date_end=test_end,
            label=f"WFO {year+1}",
        )

        # MA 基准（同期，不需要 ML 模型）
        ma_sim = simulate_ma_baseline(
            test_panel, idx_close, cfg,
            date_start=test_start_avail, date_end=test_end,
            label=f"MA基准 {year+1}",
        )

        results.append(dict(
            year=year + 1,
            port_ann=sim.get('port_ann', np.nan),
            port_sharpe=sim.get('port_sharpe', np.nan),
            port_maxdd=sim.get('port_maxdd', np.nan),
            bh_ann=sim.get('bh_ann', np.nan),
            bh_sharpe=sim.get('bh_sharpe', np.nan),
            auc=cls.get('auc', np.nan),
            ma_ann=ma_sim.get('port_ann', np.nan),
            ma_sharpe=ma_sim.get('port_sharpe', np.nan),
            ma_maxdd=ma_sim.get('port_maxdd', np.nan),
            sim=sim,
            ma_sim=ma_sim,
        ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. 可视化 & 保存
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    test_sim: dict,
    wfo_results: List[dict],
    prob_all: pd.Series,
    cfg: Config,
    ma_sim: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.join(cfg.output_dir, 'images'), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('沪深300指数择时模型评估', fontsize=14, fontweight='bold')

    # ① 测试集累计净值（ML 择时 vs MA基准 vs 买持）
    ax = axes[0, 0]
    if test_sim and 'port_rets' in test_sim:
        cum_p  = np.cumprod(1 + test_sim['port_rets'])
        cum_bh = np.cumprod(1 + test_sim['bh_rets'])
        ax.plot(cum_p,  label=f"ML择时 ({test_sim['port_ann']:.1%}/yr, Sharpe={test_sim['port_sharpe']:.2f})",
                color='steelblue')
        if ma_sim and 'port_rets' in ma_sim:
            cum_ma = np.cumprod(1 + ma_sim['port_rets'])
            ax.plot(cum_ma, label=f"MA基准 ({ma_sim['port_ann']:.1%}/yr, Sharpe={ma_sim['port_sharpe']:.2f})",
                    color='green', linestyle='-.')
        ax.plot(cum_bh, label=f"买持CSI300 ({test_sim['bh_ann']:.1%}/yr)",
                linestyle='--', alpha=0.7, color='orange')
        ax.axhline(1.0, color='gray', linewidth=0.5)
        ax.set_title('测试集累计净值（2023-2025）')
        ax.legend(fontsize=8)
        ax.set_ylabel('净值')

    # ② WFO 年度收益对比（ML / MA基准 / 买持）
    ax = axes[0, 1]
    if wfo_results:
        years   = [r['year'] for r in wfo_results]
        p_anns  = [r['port_ann'] * 100 for r in wfo_results]
        bh_anns = [r['bh_ann']  * 100 for r in wfo_results]
        ma_anns = [r.get('ma_ann', float('nan')) * 100 for r in wfo_results]
        x = np.arange(len(years))
        has_ma = any(np.isfinite(v) for v in ma_anns)
        if has_ma:
            ax.bar(x - 0.27, p_anns,  0.27, label='ML择时',  color='steelblue')
            ax.bar(x,         ma_anns, 0.27, label='MA基准',  color='green',  alpha=0.8)
            ax.bar(x + 0.27, bh_anns, 0.27, label='买持',    color='orange', alpha=0.7)
        else:
            ax.bar(x - 0.2, p_anns,  0.4, label='择时',  color='steelblue')
            ax.bar(x + 0.2, bh_anns, 0.4, label='买持',  color='orange', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_title('WFO 年度收益率（%）')
        ax.legend(fontsize=8)
        ax.set_ylabel('%')

    # ③ 测试集仓位时序
    ax = axes[1, 0]
    if test_sim and 'slot_counts' in test_sim:
        slots = np.array(test_sim['slot_counts'])
        ax.fill_between(range(len(slots)), slots, alpha=0.5, color='green', step='post')
        ax.set_title('测试集仓位档位（槽位数）')
        ax.set_ylabel('槽位数')
        ax.set_yticks([0, 10, 20])
        ax.set_ylim(-1, 22)

    # ④ pred_prob 分布
    ax = axes[1, 1]
    test_prob = prob_all[prob_all.index >= cfg.test_start]
    if len(test_prob) > 0:
        ax.hist(test_prob.values, bins=30, edgecolor='black', alpha=0.7, color='royalblue')
        ax.axvline(cfg.threshold_full, color='red',    linestyle='--',
                   label=f'满仓阈值={cfg.threshold_full}')
        ax.axvline(cfg.threshold_half, color='orange', linestyle='--',
                   label=f'半仓阈值={cfg.threshold_half}')
        ax.set_title('测试集 pred_prob 分布')
        ax.set_xlabel('pred_prob')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(cfg.output_dir, 'images', 'index_timing_eval.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表已保存: {out}")


def save_predictions(
    panel: pd.DataFrame,
    prob_all: pd.Series,
    cfg: Config,
) -> None:
    """保存每个交易日的 pred_prob 和槽位推荐（供选股模型读取）"""
    os.makedirs(os.path.join(cfg.output_dir, 'csv'), exist_ok=True)

    def _slots_with_ma(trade_date: str, prob: float) -> int:
        s = prob_to_slots(prob, cfg)
        if trade_date not in panel.index or cfg.ma_override == 'none':
            return s
        ma20_dev = panel.at[trade_date, 'close_vs_ma20'] if 'close_vs_ma20' in panel.columns else np.nan
        ma60_dev = panel.at[trade_date, 'close_vs_ma60'] if 'close_vs_ma60' in panel.columns else np.nan
        if not np.isfinite(ma20_dev): ma20_dev = 0.0
        if not np.isfinite(ma60_dev): ma60_dev = 0.0
        if cfg.ma_override == 'hard3':
            if ma20_dev < 0: return cfg.slots_empty
            elif ma60_dev < 0: return min(s, cfg.slots_half)
        elif cfg.ma_override == 'ma20only':
            if ma20_dev < 0: return cfg.slots_empty
        elif cfg.ma_override == 'soft':
            if ma20_dev < -0.03 and prob < 0.40: return cfg.slots_empty
            elif ma60_dev < 0 and prob < cfg.threshold_half: return min(s, cfg.slots_half)
        return s

    out = pd.DataFrame({
        'trade_date': prob_all.index,
        'pred_prob':  prob_all.values,
        'slots':      [_slots_with_ma(d, p) for d, p in zip(prob_all.index, prob_all.values)],
    })
    # 附加实际标签（有则附加）
    out = out.merge(
        panel[['label', 'fwd_ret']].reset_index().rename(columns={'index': 'trade_date'}),
        on='trade_date', how='left',
    )
    path = os.path.join(cfg.output_dir, 'csv', f'index_timing_predictions{cfg.pred_suffix}.csv')
    out.to_csv(path, index=False)
    print(f"  预测结果已保存: {path} ({len(out)} 行)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='沪深300指数择时模型')
    parser.add_argument('--rebal_freq',     type=int,   default=15,
                        help='持有期（交易日），与选股模型对齐，默认15')
    parser.add_argument('--threshold_full', type=float, default=0.60,
                        help='满仓阈值（pred_prob ≥ 此值 → 20槽），默认0.60')
    parser.add_argument('--threshold_half', type=float, default=0.45,
                        help='半仓阈值（pred_prob ≥ 此值 → 10槽），默认0.45')
    parser.add_argument('--slots_full',     type=int,   default=20)
    parser.add_argument('--slots_half',     type=int,   default=10)
    parser.add_argument('--no_wfo',         action='store_true',
                        help='跳过WFO，仅评估测试集（快速模式）')
    parser.add_argument('--use_rolling_pct', action='store_true',
                        help='启用滚动百分位仓位映射（解决概率跨期漂移问题）')
    parser.add_argument('--pct_full',       type=float, default=0.70,
                        help='满仓百分位阈值（--use_rolling_pct 生效时），默认0.70')
    parser.add_argument('--pct_half',       type=float, default=0.35,
                        help='半仓百分位阈值（--use_rolling_pct 生效时），默认0.35')
    # 实验参数
    parser.add_argument('--label_type',     type=str,   default='direction',
                        choices=['direction', 'ma60_state', 'meta_label'],
                        help='预测目标: direction(方向) | ma60_state(MA60位置状态) | meta_label(元标签，LdP方法)')
    parser.add_argument('--n_estimators',   type=int,   default=150,
                        help='XGBoost树数量，默认150（实验极简模型时用50）')
    parser.add_argument('--max_depth',      type=int,   default=2,
                        help='XGBoost树深度，默认2（实验极简模型时用1）')
    parser.add_argument('--ma_override',    type=str,   default='hard3',
                        choices=['hard3', 'ma20only', 'soft', 'none'],
                        help='MA覆盖模式: hard3(强三档，默认) | ma20only(仅MA20保护，配合ma60_state) | soft(软覆盖) | none(纯ML)')
    parser.add_argument('--val_year',       type=int,   default=2022,
                        help='验证集年份（默认2022）。meta_label建议用非熊市年如2021，'
                             '因2022熊市导致MA活跃样本仅79个，val_AUC极不稳定。')
    parser.add_argument('--prod',           action='store_true',
                        help='生产模式：使用 2016-2025 全量数据训练，预测存至 '
                             'index_timing_predictions_prod.csv，跳过 test 评估')
    args = parser.parse_args()

    # 生产模式：使用 2025 年数据作为 ES val，其余全作为训练，跳过 test 评估
    if args.prod:
        cfg = Config(
            rebal_freq      = args.rebal_freq,
            threshold_full  = args.threshold_full,
            threshold_half  = args.threshold_half,
            slots_full      = args.slots_full,
            slots_half      = args.slots_half,
            use_rolling_pct = args.use_rolling_pct,
            pct_full        = args.pct_full,
            pct_half        = args.pct_half,
            label_type      = args.label_type,
            n_estimators    = args.n_estimators,
            max_depth       = args.max_depth,
            ma_override     = args.ma_override,
            val_cutoff      = "20250101",   # 2025 作为 ES val
            val_end         = "20251231",
            test_start      = "20270101",   # 超出数据范围 → 跳过 test 评估
            pred_suffix     = "_prod",
        )
    else:
        cfg = Config(
            rebal_freq      = args.rebal_freq,
            threshold_full  = args.threshold_full,
            threshold_half  = args.threshold_half,
            slots_full      = args.slots_full,
            slots_half      = args.slots_half,
            use_rolling_pct = args.use_rolling_pct,
            pct_full        = args.pct_full,
            pct_half        = args.pct_half,
            label_type      = args.label_type,
            n_estimators    = args.n_estimators,
            max_depth       = args.max_depth,
            ma_override     = args.ma_override,
            val_cutoff      = f"{args.val_year}0101",
            val_end         = f"{args.val_year}1231",
        )

    print("=" * 65)
    print("沪深300指数择时模型")
    print(f"  标签: {cfg.rebal_freq}日前向收益方向  "
          f"仓位: {cfg.slots_full}/{cfg.slots_half}/0 槽")
    if cfg.use_rolling_pct:
        print(f"  仓位模式: 滚动百分位  满仓>{cfg.pct_full:.0%}  半仓>{cfg.pct_half:.0%}"
              f"  窗口={cfg.pct_window}期  预热={cfg.pct_warmup}期")
    else:
        print(f"  仓位模式: 绝对阈值  满仓阈值={cfg.threshold_full}  半仓阈值={cfg.threshold_half}")
    ma_desc = {
        'hard3':    'close<MA20→空; MA20-MA60→半; MA60+→ML自由',
        'ma20only': 'close<MA20→空; MA20+→ML自由（取消MA60半仓限制，适合ma60_state）',
        'soft':     'ML低置信+MA深跌→空/半; 其余ML直接决定（适合ma60_state标签）',
        'none':     '关闭（纯ML决策）',
    }.get(cfg.ma_override, cfg.ma_override)
    print(f"  MA覆盖: {cfg.ma_override} — {ma_desc}")
    print("=" * 65)

    # ── 1. 构建特征面板 ────────────────────────────────────────────────
    print("\n[1/4] 构建特征面板...")
    panel, feat_cols, idx_close = build_panel(cfg)

    # ── 2. 训练主模型 ──────────────────────────────────────────────────
    print("\n[2/4] 训练主模型（训练至2021，验证2022，测试2023+）...")
    all_dates = sorted(panel.index.tolist())
    val_start_avail = next((d for d in all_dates if d >= cfg.val_cutoff), cfg.val_cutoff)

    mdl = train_model(
        panel, feat_cols, cfg,
        val_start=val_start_avail,
        val_end=cfg.val_end,
        label="主模型",
    )

    # 全量预测（供保存和可视化）
    valid_mask = panel[feat_cols].notna().all(axis=1)
    prob_all   = pd.Series(
        mdl.predict_proba(panel.loc[valid_mask, feat_cols])[:, 1],
        index=panel.index[valid_mask],
    )

    # ── 3. 测试集评估 ──────────────────────────────────────────────────
    print("\n[3/4] 测试集评估...")
    # 分类评估面板：仅保留有标签的行（meta_label 下 = MA 活跃期）
    test_panel_cls = panel[panel.index >= cfg.test_start].dropna(subset=['label'])

    if len(test_panel_cls) == 0:
        print("  警告：测试集（有标签）为空，跳过评估，直接保存预测")
        save_predictions(panel, prob_all, cfg)
        return

    # 模拟面板：需要全部测试日（含 close<MA20 的空仓期）
    # meta_label 下若只用有标签的行，调仓时间表会跳过 MA 平仓区间 → 净值严重失真
    if cfg.label_type == 'meta_label':
        test_panel_sim = panel[
            (panel.index >= cfg.test_start) &
            panel[feat_cols[:3]].notna().all(axis=1)
        ]
    else:
        test_panel_sim = test_panel_cls

    test_prob_cls = prob_all.reindex(test_panel_cls.index).dropna()
    test_prob_sim = prob_all.reindex(test_panel_sim.index).dropna()

    cls_metrics = evaluate_classification(
        test_panel_cls, test_prob_cls,
        date_start=cfg.test_start,
        label="测试集",
    )
    test_sim = simulate_timing(
        test_prob_sim, idx_close,
        cfg, date_start=cfg.test_start,
        panel=test_panel_sim,
        label="测试集(ML)",
    )
    test_ma_sim = simulate_ma_baseline(
        test_panel_sim, idx_close, cfg,
        date_start=cfg.test_start,
        label="测试集(MA基准)",
    )

    # ── 4. WFO ────────────────────────────────────────────────────────
    if not args.no_wfo:
        print("\n[4/4] 滚动样本外验证（WFO 2022-2025）...")
        wfo_results = walk_forward_evaluation(panel, feat_cols, idx_close, cfg)
    else:
        print("\n[4/4] 跳过WFO（--no_wfo）")
        wfo_results = []

    # ── 5. 汇总报告 ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("最终结果汇总")
    print("=" * 65)
    print(f"\n测试集（{cfg.test_start} ~ 2025）:")
    print(f"  分类指标: AUC={cls_metrics.get('auc', 0):.3f}  "
          f"Brier={cls_metrics.get('brier', 0):.3f}  "
          f"Acc={cls_metrics.get('accuracy', 0):.1%}")
    print(f"\n  {'策略':^12}  {'年化':>9}  {'Sharpe':>7}  {'MaxDD':>9}  {'满仓%':>6}  {'半仓%':>6}  {'空仓%':>6}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*6}  {'-'*6}")
    for sim, lbl in [(test_ma_sim, 'MA基准'), (test_sim, 'ML择时'), ]:
        if sim:
            print(f"  {lbl:^12}  {sim.get('port_ann',0):>9.2%}  "
                  f"{sim.get('port_sharpe',0):>7.3f}  {sim.get('port_maxdd',0):>9.2%}  "
                  f"{sim.get('full_pct',0):>6.1%}  {sim.get('half_pct',0):>6.1%}  "
                  f"{sim.get('empty_pct',0):>6.1%}")
    bh_ann = test_sim.get('bh_ann', 0) if test_sim else 0
    bh_sh  = test_sim.get('bh_sharpe', 0) if test_sim else 0
    bh_dd  = test_sim.get('bh_maxdd', 0) if test_sim else 0
    print(f"  {'买持CSI300':^12}  {bh_ann:>9.2%}  {bh_sh:>7.3f}  {bh_dd:>9.2%}")

    if wfo_results:
        print(f"\nWFO 滚动样本外（N={len(wfo_results)}）：")
        print(f"  {'年份':>5}  {'ML年化':>8}  {'MA基准':>8}  {'买持':>8}  {'ML_Sharpe':>10}  {'ML_DD':>8}  {'MA_DD':>8}  {'AUC':>6}")
        for r in wfo_results:
            print(f"  {r['year']:>5}  {r['port_ann']:>8.2%}  "
                  f"{r.get('ma_ann', float('nan')):>8.2%}  "
                  f"{r['bh_ann']:>8.2%}  "
                  f"{r['port_sharpe']:>10.3f}  "
                  f"{r['port_maxdd']:>8.2%}  "
                  f"{r.get('ma_maxdd', float('nan')):>8.2%}  "
                  f"{r['auc']:>6.3f}")
        valid_s  = [r['port_sharpe'] for r in wfo_results if np.isfinite(r['port_sharpe'])]
        mean_s   = np.mean(valid_s) if valid_s else np.nan
        ma_anns  = [r.get('ma_ann', np.nan) for r in wfo_results]
        ml_wins  = sum(1 for r in wfo_results
                       if r['port_ann'] > r.get('ma_ann', -np.inf))
        pos_pct  = sum(1 for r in wfo_results if r['port_ann'] > 0) / len(wfo_results)
        print(f"\n  WFO 均值 Sharpe={mean_s:.3f}  年度正收益={pos_pct:.0%}  "
              f"ML胜MA基准={ml_wins}/{len(wfo_results)}")

    # 特征重要性
    importance = pd.Series(
        mdl.feature_importances_, index=feat_cols
    ).sort_values(ascending=False)
    print("\n  Top 10 特征重要性:")
    for name, imp in importance.head(10).items():
        print(f"    {name:<28} {imp:.4f}")

    # ── 6. 保存 ────────────────────────────────────────────────────────
    save_predictions(panel, prob_all, cfg)
    plot_results(test_sim, wfo_results, prob_all, cfg, ma_sim=test_ma_sim)

    imp_path = os.path.join(cfg.output_dir, 'csv', 'index_timing_feature_importance.csv')
    importance.reset_index().to_csv(imp_path, index=False, header=['feature', 'importance'])
    print(f"  特征重要性已保存: {imp_path}")

    print("\n完成。")
    print(f"  择时预测: output/csv/index_timing_predictions.csv")
    print(f"  评估图表: output/images/index_timing_eval.png")
    print()
    print("  用法提示：将 index_timing_predictions.csv 中的 'slots' 列")
    print("  读入增强型基本面策略，替换固定的 top_n 参数，实现动态仓位控制。")


if __name__ == '__main__':
    main()
