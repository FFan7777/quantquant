#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 时序择时模型（Triple-Barrier Method）

核心设计：
  标签: 三重屏障法（动态波动率设置上/下/时间屏障）→ 二分类 {0, 1}
        1 = 先触及上屏障（上涨止盈，做多信号有效）
        0 = 先触及下屏障或时间到期（止损或震荡，避免持仓）

特征工程（严格PIT，共42个因子）：
  技术面  (12): 多尺度收益率/波动率/MA偏离/RSI/布林带位置
  估值流动性(5): PE/PB/对数市值/换手率/量比
  资金流向  (4): 1/5/10日净流量/市值，大单净比例
  基本面PIT (8): ROE/ROA/毛利率/负债率/流动比/F-Score/营收净利增长
  市场宽度  (5): 等权市场收益/波动率/上涨比例/涨跌家数比
  宏观北向  (4): 北向净流入/PMI/M2增速/SHIBOR
  时间编码  (4): 月份/星期 sin-cos 变换

损失函数（可配置组合）：
  FocalLoss        : 聚焦难分样本，γ 控制难易权重差异
  AsymmetricLoss   : 非对称惩罚，假阳性（买错亏钱）> 假阴性（踏空）
  DirectionalLoss  : 对高置信度方向错误施加额外惩罚
  CombinedObjective: 上述三种损失的加权组合（通过 LossConfig 配置）
  SharpeEvalMetric : 夏普比率作为验证集早停指标

评测体系：
  算法层面 : Accuracy / Precision / Recall / F1 / AUC / 混淆矩阵
  金融层面 : 年化收益 / 最大回撤 / 夏普 / 卡玛 / 胜率 / 盈亏比
  鲁棒性   : 滚动样本外 WFO（每年一个测试窗口）/ 参数敏感性分析

使用方式:
  python xgboost_market_timing.py
  python xgboost_market_timing.py --profit_take 2.0 --stop_loss 0.8
  python xgboost_market_timing.py --focal_weight 0.6 --fp_penalty 3.0
"""

import os
import sys
import time
import warnings
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
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
class LossConfig:
    """损失函数组合配置（权重自动归一化）"""
    # 各组件权重
    focal_weight: float = 0.40       # Focal Loss 权重
    asymmetric_weight: float = 0.40  # 非对称惩罚 权重
    direction_weight: float = 0.20   # 方向准确率 权重

    # Focal Loss 参数
    focal_gamma: float = 2.0         # γ 越大越聚焦难样本

    # 非对称损失参数
    fp_penalty: float = 1.5          # 假阳性惩罚倍数（买错 → 真实亏钱）
    fn_penalty: float = 1.0          # 假阴性惩罚倍数（踏空 → 只是错过）

    # 方向准确率损失参数
    direction_penalty: float = 2.0   # 高置信度方向错误的额外惩罚系数

    def __post_init__(self):
        total = self.focal_weight + self.asymmetric_weight + self.direction_weight
        if total > 0:
            self.focal_weight /= total
            self.asymmetric_weight /= total
            self.direction_weight /= total


@dataclass
class ModelConfig:
    """完整模型配置"""
    # ─── 三重屏障 ───────────────────────────────────────────────────────────
    profit_take: float = 1.0      # 上屏障 = entry × (1 + profit_take × σ_hold)
    stop_loss: float = 1.0        # 下屏障 = entry × (1 - stop_loss   × σ_hold)
    max_hold: int = 15             # 垂直屏障：最大持有期（交易日）
    vol_lookback: int = 20         # 计算 σ 的滚动窗口（年化，内部转持有期）

    # ─── 数据 ───────────────────────────────────────────────────────────────
    db_path: str = "data/quant.duckdb"
    benchmark: str = "000300.SH"
    data_start: str = "20160101"
    train_start: str = "20180101"
    train_end: str = "20221231"
    val_cutoff: str = "20220101"   # 训练集内：验证集起点
    test_start: str = "20230201"   # 样本外测试起点（保守隔离 20 交易日）
    end_date: str = "20251231"
    rebal_freq: int = 15           # 调仓频率（交易日），与 max_hold=15 对齐消除标签错位
    min_mktcap: float = 10.0       # 最小市值过滤（亿元）

    # ─── XGBoost ────────────────────────────────────────────────────────────
    n_estimators: int = 1000
    max_depth: int = 4
    learning_rate: float = 0.02
    subsample: float = 0.70
    colsample_bytree: float = 0.70
    min_child_weight: int = 10    # 降低（原30对7%正样本太严格，导致正样本叶节点无法分裂）
    reg_lambda: float = 5.0
    reg_alpha: float = 0.5
    early_stopping_rounds: int = 50

    # ─── 组合模拟 ────────────────────────────────────────────────────────────
    top_n: int = 30                # 最多持有股票数
    signal_threshold: float = 0.50 # P(label=1) > threshold → 买入信号
    min_breadth: float = 0.0       # 市场宽度门槛：breadth_pct_ma20 < 此值时暂停新开仓（0=不过滤）
    commission: float = 0.0003     # 双边佣金
    stamp_tax: float = 0.001       # 印花税（仅卖出）
    slippage: float = 0.001        # 双边滑点

    # ─── 损失函数 ────────────────────────────────────────────────────────────
    loss: LossConfig = field(default_factory=LossConfig)

    # ─── 输出 ────────────────────────────────────────────────────────────────
    output_dir: str = "output"


# 特征列定义
TECH_COLS   = ['ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
               'vol_5d', 'vol_20d', 'close_vs_ma20', 'close_vs_ma60',
               'rsi_14', 'bb_pos']
VAL_COLS    = ['pe_ttm', 'pb', 'log_mktcap', 'turnover_20d', 'volume_ratio']
MF_COLS     = ['mf_1d_mv', 'mf_5d_mv', 'mf_10d_mv', 'large_net_5d_ratio']
FUND_COLS   = ['roe_ann', 'roa', 'gross_margin', 'debt_ratio',
               'current_ratio', 'fscore', 'rev_growth_yoy', 'ni_growth_yoy']
BREADTH_COLS = ['mkt_ret_1d', 'mkt_ret_5d', 'mkt_vol_20d',
                'breadth_pct_ma20', 'advance_decline']
MACRO_COLS  = ['hsgt_net_flow', 'pmi_mfg', 'm2_yoy', 'shibor_3m']
TIME_COLS   = ['month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']

ALL_FEATURES = (TECH_COLS + VAL_COLS + MF_COLS + FUND_COLS +
                BREADTH_COLS + MACRO_COLS + TIME_COLS)   # 共 42 个


# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据库连接
# ══════════════════════════════════════════════════════════════════════════════

def get_conn(cfg: ModelConfig, read_only: bool = True):
    return duckdb.connect(cfg.db_path, read_only=read_only)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 特征工程
# ══════════════════════════════════════════════════════════════════════════════

def load_price_matrices(conn, cfg: ModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载前复权收盘价矩阵，并计算滚动年化波动率矩阵。
    返回: (close_pivot, vol_pivot) ── 均为 date × ts_code
    """
    print("  加载价格矩阵（前复权）...")
    df = conn.execute(f"""
        SELECT trade_date, ts_code, close, pct_chg
        FROM daily_price
        WHERE trade_date >= '{cfg.data_start}' AND trade_date <= '{cfg.end_date}'
          AND ts_code NOT LIKE '8%'
          AND ts_code NOT LIKE '4%'
        ORDER BY trade_date, ts_code
    """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str)

    close_pivot = df.pivot(index='trade_date', columns='ts_code', values='close')
    pctchg_pivot = df.pivot(index='trade_date', columns='ts_code', values='pct_chg') / 100.0

    # 滚动年化波动率（基于日收益率）
    vol_pivot = pctchg_pivot.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback // 2).std() * np.sqrt(252)

    print(f"    价格矩阵: {close_pivot.shape[0]} 天 × {close_pivot.shape[1]} 只")
    return close_pivot, vol_pivot


def load_tech_val_features(conn, cfg: ModelConfig) -> pd.DataFrame:
    """
    技术面 + 估值特征（SQL 窗口函数，一次性加载）
    返回: (ts_code, trade_date) → 17 个特征
    """
    print("  加载技术/估值特征（SQL 窗口函数）...")
    df = conn.execute(f"""
        WITH dp AS (
            SELECT ts_code, trade_date, close, pct_chg, vol
            FROM daily_price
            WHERE trade_date >= '{cfg.data_start}' AND trade_date <= '{cfg.end_date}'
              AND ts_code NOT LIKE '8%'
              AND ts_code NOT LIKE '4%'
        ),
        tech AS (
            SELECT
                ts_code, trade_date, close,
                close / NULLIF(LAG(close,  1) OVER w, 0) - 1  AS ret_1d,
                close / NULLIF(LAG(close,  3) OVER w, 0) - 1  AS ret_3d,
                close / NULLIF(LAG(close,  5) OVER w, 0) - 1  AS ret_5d,
                close / NULLIF(LAG(close, 10) OVER w, 0) - 1  AS ret_10d,
                close / NULLIF(LAG(close, 20) OVER w, 0) - 1  AS ret_20d,
                close / NULLIF(LAG(close, 60) OVER w, 0) - 1  AS ret_60d,
                STDDEV(pct_chg/100.0) OVER (w ROWS BETWEEN  4 PRECEDING AND CURRENT ROW)
                    * SQRT(252)   AS vol_5d,
                STDDEV(pct_chg/100.0) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                    * SQRT(252)   AS vol_20d,
                AVG(close)        OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
                AVG(close)        OVER (w ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60,
                -- Bollinger Band (20, 2σ)
                AVG(close)        OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS bb_mid,
                STDDEV(close)     OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS bb_std
            FROM dp
            WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
        ),
        db AS (
            SELECT
                ts_code, trade_date,
                pe_ttm, pb,
                LN(NULLIF(total_mv, 0))  AS log_mktcap,
                total_mv / 10000.0       AS total_mv_100m,
                AVG(turnover_rate) OVER (
                    PARTITION BY ts_code ORDER BY trade_date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                )                        AS turnover_20d,
                volume_ratio
            FROM daily_basic
            WHERE trade_date >= '{cfg.data_start}' AND trade_date <= '{cfg.end_date}'
              AND total_mv > 0
        )
        SELECT
            t.ts_code, t.trade_date,
            t.ret_1d, t.ret_3d, t.ret_5d, t.ret_10d, t.ret_20d, t.ret_60d,
            t.vol_5d, t.vol_20d,
            t.close / NULLIF(t.ma20, 0) - 1  AS close_vs_ma20,
            t.close / NULLIF(t.ma60, 0) - 1  AS close_vs_ma60,
            -- Bollinger Band position: 0 = 下轨, 1 = 上轨
            CASE WHEN t.bb_std > 0
                 THEN (t.close - (t.bb_mid - 2*t.bb_std)) /
                      NULLIF(4 * t.bb_std, 0)
                 ELSE 0.5 END  AS bb_pos,
            d.pe_ttm, d.pb, d.log_mktcap, d.total_mv_100m,
            d.turnover_20d, d.volume_ratio
        FROM tech t
        JOIN db d ON t.ts_code = d.ts_code AND t.trade_date = d.trade_date
        WHERE t.ret_60d IS NOT NULL
          AND d.log_mktcap IS NOT NULL
    """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str)
    print(f"    技术/估值特征: {len(df):,} 行, {df['ts_code'].nunique()} 只股票")
    return df


def compute_rsi(close_pivot: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder RSI，全矩阵向量化"""
    delta    = close_pivot.diff(1)
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_moneyflow_features(conn, cfg: ModelConfig) -> pd.DataFrame:
    """
    资金流向特征（SQL 窗口函数）
    返回: (ts_code, trade_date) → [mf_1d_raw, mf_5d_raw, mf_10d_raw,
                                    large_net_5d, total_flow_5d]
    """
    print("  加载资金流向特征...")
    df = conn.execute(f"""
        SELECT
            ts_code, trade_date,
            net_mf_amount  AS mf_1d_raw,
            SUM(net_mf_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN  4 PRECEDING AND CURRENT ROW
            )  AS mf_5d_raw,
            SUM(net_mf_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN  9 PRECEDING AND CURRENT ROW
            )  AS mf_10d_raw,
            SUM(buy_lg_amount + buy_elg_amount
                - sell_lg_amount - sell_elg_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )  AS large_net_5d,
            SUM(buy_sm_amount + buy_md_amount + buy_lg_amount + buy_elg_amount
                + sell_sm_amount + sell_md_amount + sell_lg_amount + sell_elg_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )  AS total_flow_5d
        FROM moneyflow
        WHERE trade_date >= '{cfg.data_start}' AND trade_date <= '{cfg.end_date}'
    """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str)
    print(f"    资金流向: {len(df):,} 行")
    return df


def load_fundamental_panel(conn) -> pd.DataFrame:
    """
    从三张报表（IS + BS + CF）构建 PIT 基本面面板。
    使用 f_ann_date（第一披露日）确保无未来信息。
    """
    print("  加载基本面（PIT，三表合并）...")
    df = conn.execute("""
        WITH is_t AS (
            SELECT ts_code, end_date,
                   MIN(COALESCE(f_ann_date, ann_date)) AS f_ann_date,
                   FIRST(revenue            ORDER BY ann_date DESC) AS revenue,
                   FIRST(oper_cost          ORDER BY ann_date DESC) AS oper_cost,
                   FIRST(n_income_attr_p    ORDER BY ann_date DESC) AS n_income,
                   FIRST(basic_eps          ORDER BY ann_date DESC) AS eps
            FROM income_statement WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        ),
        bs_t AS (
            SELECT ts_code, end_date,
                   FIRST(total_assets               ORDER BY ann_date DESC) AS total_assets,
                   FIRST(total_liab                 ORDER BY ann_date DESC) AS total_liab,
                   FIRST(total_hldr_eqy_inc_min_int ORDER BY ann_date DESC) AS equity,
                   FIRST(total_cur_assets           ORDER BY ann_date DESC) AS cur_assets,
                   FIRST(total_cur_liab             ORDER BY ann_date DESC) AS cur_liab
            FROM balance_sheet WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        ),
        cf_t AS (
            SELECT ts_code, end_date,
                   FIRST(n_cashflow_act ORDER BY ann_date DESC) AS ocf
            FROM cash_flow WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        )
        SELECT i.ts_code, i.end_date, i.f_ann_date,
               i.revenue, i.oper_cost, i.n_income,
               b.total_assets, b.total_liab, b.equity,
               b.cur_assets, b.cur_liab, c.ocf
        FROM is_t i
        LEFT JOIN bs_t b USING (ts_code, end_date)
        LEFT JOIN cf_t c USING (ts_code, end_date)
        ORDER BY ts_code, end_date
    """).fetchdf()

    ta  = df['total_assets'].replace(0, np.nan)
    eq  = df['equity'].replace(0, np.nan)
    rev = df['revenue'].replace(0, np.nan)
    cl  = df['cur_liab'].replace(0, np.nan)

    df['period_months'] = df['end_date'].astype(str).str[4:6].map(
        {'03': 3, '06': 6, '09': 9, '12': 12}).fillna(12).astype(int)
    pm = 12.0 / df['period_months']

    df['roe_raw']      = df['n_income'] / eq * 100.0
    df['roe_ann']      = df['roe_raw'] * pm
    df['roa']          = df['n_income'] / ta * pm * 100.0
    df['gross_margin'] = (df['revenue'] - df['oper_cost']) / rev * 100.0
    df['debt_ratio']   = df['total_liab'] / ta
    df['current_ratio']= df['cur_assets'] / cl
    df['assets_turn']  = df['revenue'] / ta

    # YoY F-Score
    df['end_date'] = df['end_date'].astype(str)
    dd = df.drop_duplicates(subset=['ts_code', 'end_date'], keep='last').copy()
    dd['year']          = dd['end_date'].str[:4].astype(int)
    dd['mmdd']          = dd['end_date'].str[4:]
    dd['prev_end_date'] = (dd['year'] - 1).astype(str) + dd['mmdd']

    prev = dd[['ts_code', 'end_date', 'roa', 'debt_ratio', 'current_ratio',
               'gross_margin', 'assets_turn', 'n_income', 'revenue']].rename(columns={
        'end_date': 'prev_end_date', 'roa': 'roa_p', 'debt_ratio': 'da_p',
        'current_ratio': 'cr_p', 'gross_margin': 'gpm_p', 'assets_turn': 'at_p',
        'n_income': 'ni_p', 'revenue': 'rev_p'})
    dd = dd.merge(prev, on=['ts_code', 'prev_end_date'], how='left')

    roa_v = dd['roa'].fillna(0)
    ocf_v = dd['ocf'].fillna(0)
    ta_v  = dd['total_assets'].replace(0, np.nan)

    dd['f1'] = (roa_v > 0).astype(int)
    dd['f2'] = (ocf_v > 0).astype(int)
    dd['f3'] = (dd['n_income'].notna() & dd['ni_p'].notna() &
                (dd['n_income'] > dd['ni_p'])).astype(int)
    dd['f4'] = (ocf_v / ta_v > roa_v).astype(int)
    dd['f5'] = (dd['debt_ratio'].notna() & dd['da_p'].notna() &
                (dd['debt_ratio'] < dd['da_p'])).astype(int)
    dd['f6'] = (dd['current_ratio'].notna() & dd['cr_p'].notna() &
                (dd['current_ratio'] > dd['cr_p'])).astype(int)
    dd['f7'] = (dd['revenue'].notna() & dd['rev_p'].notna() &
                (dd['revenue'] > dd['rev_p'])).astype(int)
    dd['f8'] = (dd['gross_margin'].notna() & dd['gpm_p'].notna() &
                (dd['gross_margin'] > dd['gpm_p'])).astype(int)
    dd['f9'] = (dd['assets_turn'].notna() & dd['at_p'].notna() &
                (dd['assets_turn'] > dd['at_p'])).astype(int)
    dd['fscore'] = dd[['f1','f2','f3','f4','f5','f6','f7','f8','f9']].sum(axis=1)

    dd['rev_growth_yoy'] = np.where(
        dd['rev_p'].notna() & (dd['rev_p'] != 0),
        (dd['revenue'] - dd['rev_p']) / dd['rev_p'].abs(), np.nan)
    dd['ni_growth_yoy'] = np.where(
        dd['ni_p'].notna() & (dd['ni_p'] > 0),
        (dd['n_income'] - dd['ni_p']) / dd['ni_p'].abs(), np.nan)

    keep = ['ts_code', 'end_date', 'fscore', 'rev_growth_yoy', 'ni_growth_yoy']
    df = df.merge(dd[keep], on=['ts_code', 'end_date'], how='left')

    fund_cols = ['ts_code', 'f_ann_date', 'end_date',
                 'roe_ann', 'roa', 'gross_margin', 'debt_ratio',
                 'current_ratio', 'fscore', 'rev_growth_yoy', 'ni_growth_yoy']
    out = df[fund_cols].dropna(subset=['f_ann_date']).copy()
    out['f_ann_date'] = out['f_ann_date'].astype(str)
    out = out.sort_values(['ts_code', 'f_ann_date']).reset_index(drop=True)
    print(f"    基本面面板: {len(out):,} 条, {out['ts_code'].nunique()} 只股票")
    return out


def join_fundamental_pit(fund_df: pd.DataFrame, rebal_keys: pd.DataFrame) -> pd.DataFrame:
    """
    PIT join: 对每个 (ts_code, trade_date) 找最新已披露基本面。
    使用 np.searchsorted 高效实现。
    """
    print("  PIT 合并基本面...")
    keys = rebal_keys[['ts_code', 'trade_date']].copy()
    keys['_td_int'] = keys['trade_date'].str.replace('-', '').astype(int)

    fund_df = fund_df.copy()
    fund_df['_ann_int'] = fund_df['f_ann_date'].str.replace('-', '').astype(int)

    value_cols = [c for c in fund_df.columns
                  if c not in ('ts_code', 'f_ann_date', 'end_date', '_ann_int')]

    fund_grouped = {ts: grp.sort_values('_ann_int').reset_index(drop=True)
                    for ts, grp in fund_df.groupby('ts_code')}

    results = []
    for ts_code, kg in keys.groupby('ts_code'):
        kg = kg.copy()
        if ts_code not in fund_grouped:
            for col in value_cols:
                kg[col] = np.nan
            results.append(kg.drop(columns=['_td_int']))
            continue
        fg = fund_grouped[ts_code]
        ann_ints = fg['_ann_int'].values
        td_ints  = kg['_td_int'].values
        idxs = np.searchsorted(ann_ints, td_ints, side='right') - 1
        valid = idxs >= 0
        for col in value_cols:
            col_vals = fg[col].values
            kg[col] = np.where(valid, col_vals[np.maximum(idxs, 0)], np.nan)
        results.append(kg.drop(columns=['_td_int']))

    result = pd.concat(results, ignore_index=True)
    print(f"    PIT 匹配: {result['fscore'].notna().sum():,} / {len(result):,}")
    return result


def compute_market_breadth(close_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    从价格矩阵计算市场宽度特征（无未来信息，全部为历史统计）。
    返回: date → [mkt_ret_1d, mkt_ret_5d, mkt_vol_20d, breadth_pct_ma20, advance_decline]
    """
    print("  计算市场宽度特征...")
    daily_rets = close_pivot.pct_change(1)

    mkt_ret_1d  = daily_rets.mean(axis=1)
    mkt_ret_5d  = close_pivot.pct_change(5).mean(axis=1)
    mkt_vol_20d = daily_rets.rolling(20).std().mean(axis=1) * np.sqrt(252)

    ma20 = close_pivot.rolling(20).mean()
    breadth_pct_ma20 = (close_pivot >= ma20).mean(axis=1)

    n_up   = (daily_rets > 0).sum(axis=1)
    n_down = (daily_rets < 0).sum(axis=1)
    advance_decline = n_up / (n_down + 1)

    breadth = pd.DataFrame({
        'mkt_ret_1d':       mkt_ret_1d,
        'mkt_ret_5d':       mkt_ret_5d,
        'mkt_vol_20d':      mkt_vol_20d,
        'breadth_pct_ma20': breadth_pct_ma20,
        'advance_decline':  advance_decline,
    })
    breadth.index = breadth.index.astype(str)
    print(f"    市场宽度: {len(breadth)} 天")
    return breadth


def load_macro_features(conn, cfg: ModelConfig) -> pd.DataFrame:
    """
    加载宏观因子：北向资金净流入、PMI、M2同比、SHIBOR 3个月。
    月度数据前向填充到日频。
    """
    print("  加载宏观/北向特征...")

    # 北向资金（日频）
    try:
        hsgt = conn.execute(f"""
            SELECT trade_date, north_money AS hsgt_net_flow
            FROM moneyflow_hsgt
            WHERE trade_date >= '{cfg.data_start}' AND trade_date <= '{cfg.end_date}'
            ORDER BY trade_date
        """).fetchdf()
        hsgt['trade_date'] = hsgt['trade_date'].astype(str)
        hsgt = hsgt.set_index('trade_date')
    except Exception:
        hsgt = pd.DataFrame(columns=['hsgt_net_flow'])

    # PMI（月度）
    try:
        pmi = conn.execute("""
            SELECT month AS trade_date, pmi AS pmi_mfg
            FROM cn_pmi ORDER BY month
        """).fetchdf()
        pmi['trade_date'] = pmi['trade_date'].astype(str).str.replace('-', '')
        pmi = pmi.set_index('trade_date')
    except Exception:
        pmi = pd.DataFrame(columns=['pmi_mfg'])

    # M2 同比（月度）
    try:
        m2 = conn.execute("""
            SELECT month AS trade_date, m2_yoy
            FROM cn_m ORDER BY month
        """).fetchdf()
        m2['trade_date'] = m2['trade_date'].astype(str).str.replace('-', '')
        m2 = m2.set_index('trade_date')
    except Exception:
        m2 = pd.DataFrame(columns=['m2_yoy'])

    # SHIBOR 3M（日频）
    try:
        shibor = conn.execute(f"""
            SELECT date AS trade_date, m3 AS shibor_3m
            FROM shibor
            WHERE date >= '{cfg.data_start}' AND date <= '{cfg.end_date}'
            ORDER BY date
        """).fetchdf()
        shibor['trade_date'] = shibor['trade_date'].astype(str)
        shibor = shibor.set_index('trade_date')
    except Exception:
        shibor = pd.DataFrame(columns=['shibor_3m'])

    # 取全量交易日列表（用 hsgt 或 shibor）
    # 获取每个特征，然后合并，月度数据前向填充
    macro_pieces = []
    for name, df_piece in [('hsgt', hsgt), ('pmi', pmi), ('m2', m2), ('shibor', shibor)]:
        if not df_piece.empty:
            macro_pieces.append(df_piece)

    if not macro_pieces:
        print("    ⚠ 无宏观数据，跳过")
        return pd.DataFrame()

    macro = pd.concat(macro_pieces, axis=1).sort_index()
    # 月度数据按日前向填充（PMI, M2）
    macro = macro.ffill()
    print(f"    宏观/北向: {len(macro)} 天, 列: {list(macro.columns)}")
    return macro


def add_time_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间编码特征（月份和星期的 sin/cos 变换）"""
    dates = pd.to_datetime(df['trade_date'].astype(str))
    month   = dates.dt.month
    weekday = dates.dt.dayofweek  # 0=周一, 4=周五
    df['month_sin']   = np.sin(2 * np.pi * month   / 12)
    df['month_cos']   = np.cos(2 * np.pi * month   / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * weekday /  5)
    df['weekday_cos'] = np.cos(2 * np.pi * weekday /  5)
    return df


def load_stock_info(conn) -> pd.DataFrame:
    return conn.execute("""
        SELECT ts_code, COALESCE(industry, '未知') AS industry
        FROM stock_basic
    """).fetchdf()


# ══════════════════════════════════════════════════════════════════════════════
# 4. 三重屏障标签
# ══════════════════════════════════════════════════════════════════════════════

def compute_triple_barrier_labels(
    close_pivot: pd.DataFrame,
    vol_pivot: pd.DataFrame,
    cfg: ModelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    三重屏障打标（向量化，逐日扫描前向窗口）。

    对每个 (date, ts_code)：
      上屏障 = p_t × (1 + profit_take × σ_t)
      下屏障 = p_t × (1 - stop_loss   × σ_t)
      垂直屏障 = 持有满 max_hold 交易日

    返回:
      label_pivot   : date × ts_code，值 ∈ {1, 0}
                      1 = 先触及上屏障（做多成功），0 = 其余（止损或震荡）
      actual_ret_pivot: date × ts_code，实际持有收益率（垂直屏障到期时）
    """
    print(f"  计算三重屏障标签"
          f" (profit_take={cfg.profit_take}, stop_loss={cfg.stop_loss},"
          f" max_hold={cfg.max_hold})...")

    price_arr = close_pivot.values.astype(np.float64)
    vol_arr   = vol_pivot.reindex(columns=close_pivot.columns).values.astype(np.float64)
    n_dates, n_stocks = price_arr.shape

    label_arr      = np.full((n_dates, n_stocks), np.nan)
    actual_ret_arr = np.full((n_dates, n_stocks), np.nan)

    # 年化波动率 → max_hold 持有期波动率：σ_hold = σ_annual × √(max_hold/252)
    # 屏障宽度基于持有期波动率，避免以年化 σ 作为单期屏障导致屏障过宽（BUG修复）
    hold_vol_factor = np.sqrt(cfg.max_hold / 252.0)   # e.g. √(15/252) ≈ 0.244

    for i in range(n_dates - cfg.max_hold):
        p0    = price_arr[i]            # (n_stocks,)
        sigma = vol_arr[i]              # (n_stocks,) 年化波动率
        # sigma 中可能有 nan（数据不足），对应股票跳过
        valid = np.isfinite(p0) & np.isfinite(sigma) & (p0 > 0) & (sigma > 0)
        if not valid.any():
            continue

        hold_sigma = sigma * hold_vol_factor           # 持有期波动率（正确）
        upper = p0 * (1 + cfg.profit_take * hold_sigma)   # (n_stocks,)
        lower = p0 * (1 - cfg.stop_loss   * hold_sigma)   # (n_stocks,)

        # 前向价格窗口: (max_hold, n_stocks)
        fwd = price_arr[i+1 : i+cfg.max_hold+1]

        # 初始化当天标签为 0（垂直屏障 = 震荡）
        lbl = np.zeros(n_stocks)

        # 逐步扫描：哪个屏障最先被触及
        already_hit = np.zeros(n_stocks, dtype=bool)
        for k in range(cfg.max_hold):
            fwd_k = fwd[k]
            can_hit = valid & ~already_hit

            hit_upper = can_hit & (fwd_k >= upper)
            hit_lower = can_hit & (fwd_k <= lower)

            lbl[hit_upper] = 1     # 做多成功
            lbl[hit_lower] = 0     # 止损（已为 0，无需赋值）
            already_hit |= (hit_upper | hit_lower)

        # 实际收益：max_hold 日后的持有收益率
        future_price = price_arr[i + cfg.max_hold]
        actual_ret = np.where(valid, (future_price - p0) / p0, np.nan)

        # 对无效股票（p0=nan 或 sigma=nan）设为 nan
        lbl[~valid] = np.nan

        label_arr[i]      = lbl
        actual_ret_arr[i] = actual_ret

    label_pivot = pd.DataFrame(
        label_arr, index=close_pivot.index, columns=close_pivot.columns)
    actual_ret_pivot = pd.DataFrame(
        actual_ret_arr, index=close_pivot.index, columns=close_pivot.columns)

    n_labeled = np.isfinite(label_arr).sum()
    n_positive = (label_arr == 1).sum()
    pos_rate = n_positive / n_labeled if n_labeled > 0 else 0
    print(f"    标签: {n_labeled:,} 个有效, 正样本率 {pos_rate:.1%} (label=1 即触上屏障)")
    return label_pivot, actual_ret_pivot


# ══════════════════════════════════════════════════════════════════════════════
# 5. Panel 组装
# ══════════════════════════════════════════════════════════════════════════════

def get_rebal_dates(conn, cfg: ModelConfig) -> List[str]:
    df = conn.execute(f"""
        WITH dates AS (
            SELECT DISTINCT trade_date FROM daily_price
            WHERE trade_date >= '{cfg.train_start}' AND trade_date <= '{cfg.end_date}'
        ),
        numbered AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (ORDER BY trade_date) - 1 AS rn
            FROM dates
        )
        SELECT trade_date FROM numbered WHERE rn % {cfg.rebal_freq} = 0
        ORDER BY trade_date
    """).fetchdf()
    dates = df['trade_date'].astype(str).tolist()
    print(f"  调仓日: {len(dates)} 个 ({dates[0]} ~ {dates[-1]})")
    return dates


def build_timing_panel(
    tech_df:        pd.DataFrame,
    rsi_matrix:     pd.DataFrame,
    mf_df:          pd.DataFrame,
    fund_pit:       pd.DataFrame,
    breadth_df:     pd.DataFrame,
    macro_df:       pd.DataFrame,
    label_pivot:    pd.DataFrame,
    ret_pivot:      pd.DataFrame,
    port_ret_pivot: pd.DataFrame,
    stock_info:     pd.DataFrame,
    rebal_dates:    List[str],
    cfg:            ModelConfig,
) -> pd.DataFrame:
    """组装全量 Panel：特征 + 标签"""
    print("  组装 Panel...")
    rebal_set = set(rebal_dates)

    # 1. 基础：技术/估值特征（筛选调仓日）
    base = tech_df[tech_df['trade_date'].isin(rebal_set)].copy()

    # 2. RSI
    rsi_long = rsi_matrix.stack().reset_index()
    rsi_long.columns = ['trade_date', 'ts_code', 'rsi_14']
    rsi_long['trade_date'] = rsi_long['trade_date'].astype(str)
    rsi_long = rsi_long[rsi_long['trade_date'].isin(rebal_set)]
    base = base.merge(rsi_long[['ts_code', 'trade_date', 'rsi_14']],
                      on=['ts_code', 'trade_date'], how='left')

    # 3. 资金流向（归一化到市值）
    mf_rebal = mf_df[mf_df['trade_date'].isin(rebal_set)].copy()
    base = base.merge(mf_rebal, on=['ts_code', 'trade_date'], how='left')
    mv_wan = base['total_mv_100m'] * 10000  # 亿元 → 万元
    base['mf_1d_mv']           = base['mf_1d_raw']   / mv_wan.replace(0, np.nan)
    base['mf_5d_mv']           = base['mf_5d_raw']   / mv_wan.replace(0, np.nan)
    base['mf_10d_mv']          = base['mf_10d_raw']  / mv_wan.replace(0, np.nan)
    base['large_net_5d_ratio'] = (base['large_net_5d'] /
                                   base['total_flow_5d'].replace(0, np.nan))
    base = base.drop(columns=['mf_1d_raw', 'mf_5d_raw', 'mf_10d_raw',
                               'large_net_5d', 'total_flow_5d'], errors='ignore')

    # 4. 基本面 PIT
    fund_cols_keep = ['ts_code', 'trade_date'] + FUND_COLS
    base = base.merge(fund_pit[[c for c in fund_cols_keep if c in fund_pit.columns]],
                      on=['ts_code', 'trade_date'], how='left')

    # 5. 市场宽度（按日期）
    if not breadth_df.empty:
        breadth_rebal = breadth_df[breadth_df.index.isin(rebal_set)].reset_index()
        breadth_rebal.columns = ['trade_date'] + list(breadth_rebal.columns[1:])
        base = base.merge(breadth_rebal, on='trade_date', how='left')

    # 6. 宏观特征（按日期，月度数据已前向填充）
    if not macro_df.empty:
        macro_rebal = macro_df[macro_df.index.isin(rebal_set)].reset_index()
        macro_rebal.columns = ['trade_date'] + list(macro_rebal.columns[1:])
        base = base.merge(macro_rebal, on='trade_date', how='left')

    # 7. 时间编码
    base = add_time_encoding(base)

    # 8. 行业信息（用于中性化）
    base = base.merge(stock_info, on='ts_code', how='left')
    base['industry'] = base['industry'].fillna('未知')

    # 9. 标签（三重屏障）
    label_long = label_pivot.stack().reset_index()
    label_long.columns = ['trade_date', 'ts_code', 'binary_label']
    label_long['trade_date'] = label_long['trade_date'].astype(str)
    label_long = label_long[label_long['trade_date'].isin(rebal_set)]

    ret_long = ret_pivot.stack().reset_index()
    ret_long.columns = ['trade_date', 'ts_code', 'actual_ret']
    ret_long['trade_date'] = ret_long['trade_date'].astype(str)
    ret_long = ret_long[ret_long['trade_date'].isin(rebal_set)]

    # 5日持有期收益（用于组合模拟，与调仓周期匹配）
    port_ret_long = port_ret_pivot.stack().reset_index()
    port_ret_long.columns = ['trade_date', 'ts_code', 'port_ret']
    port_ret_long['trade_date'] = port_ret_long['trade_date'].astype(str)
    port_ret_long = port_ret_long[port_ret_long['trade_date'].isin(rebal_set)]

    base = base.merge(label_long,    on=['ts_code', 'trade_date'], how='inner')
    base = base.merge(ret_long,      on=['ts_code', 'trade_date'], how='left')
    base = base.merge(port_ret_long, on=['ts_code', 'trade_date'], how='left')
    base = base.dropna(subset=['binary_label'])

    # 10. 市值过滤
    before = len(base)
    base = base[base['total_mv_100m'] >= cfg.min_mktcap]
    base = base[~base['ts_code'].str.startswith(('8', '4'))].copy()
    print(f"    Panel: {len(base):,} 行 | 过滤 {before-len(base):,} 行"
          f" | {base['trade_date'].nunique()} 个截面"
          f" | 正样本率 {base['binary_label'].mean():.1%}")
    return base.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 数据预处理
# ══════════════════════════════════════════════════════════════════════════════

def winsorize_mad(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    med = s.median()
    if pd.isna(med):
        return s
    mad = (s - med).abs().median()
    if mad == 0:
        return s
    k = 1.4826 * mad
    return s.clip(lower=med - threshold * k, upper=med + threshold * k)


def neutralize_cross_section(df: pd.DataFrame, factor_cols: list,
                              industry_col: str = 'industry',
                              logmktcap_col: str = 'log_mktcap') -> pd.DataFrame:
    """截面行业 + 市值中性化（Ridge 回归残差）"""
    df = df.copy()
    inds = pd.get_dummies(df[industry_col], prefix='ind', drop_first=True, dtype=float)
    logmv = df[logmktcap_col].fillna(df[logmktcap_col].median())
    X_ctrl = pd.concat([logmv.rename('log_mktcap'), inds], axis=1).values.astype(float)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    for col in factor_cols:
        y = df[col].values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            continue
        ridge.fit(X_ctrl[valid], y[valid])
        pred = ridge.predict(X_ctrl)
        df[col] = np.where(valid, y - pred + pred[valid].mean(), np.nan)
    return df


def preprocess_panel(panel: pd.DataFrame, feature_cols: list,
                     neutralize: bool = True) -> pd.DataFrame:
    """每个截面：MAD 去极值 → 中性化 → Z-score 标准化"""
    print("  预处理: 去极值 → 中性化 → Z-score...")
    panel = panel.copy()

    skip_neutralize = {'log_mktcap', 'fscore', 'binary_label'}

    def process_group(grp):
        for col in feature_cols:
            if col in grp.columns:
                grp[col] = winsorize_mad(grp[col])
        if neutralize and len(grp) > 20:
            nc = [c for c in feature_cols if c not in skip_neutralize and c in grp.columns]
            grp = neutralize_cross_section(grp, nc)
        for col in feature_cols:
            if col not in grp.columns:
                continue
            s   = grp[col]
            mu  = s.mean()
            std = s.std()
            grp[col] = (s - mu) / std if std > 0 else 0.0
        return grp

    processed = [process_group(g.copy()) for _, g in panel.groupby('trade_date')]
    panel = pd.concat(processed, ignore_index=True)
    print(f"    预处理完成: {panel.shape}")
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# 7. 自定义损失函数
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class FocalLossObjective:
    """
    Focal Loss: 降低简单样本权重，聚焦难分类样本。
    适合解决市场震荡期大量"容易预测的 0"淹没趋势信号的问题。

    grad_i = (1-pt)^γ × (p - y)
    hess_i ≈ (1-pt)^γ × p × (1-p)  （忽略二阶项以保持 hess 正定）
    """
    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma

    def __call__(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y  = dtrain.get_label()
        p  = _sigmoid(y_pred)
        pt = np.where(y == 1, p, 1 - p)
        fw = (1 - pt) ** self.gamma          # focal weight
        grad = fw * (p - y)
        hess = np.maximum(fw * p * (1 - p), 1e-6)
        return grad, hess


class AsymmetricLossObjective:
    """
    非对称惩罚损失：对假阳性（预测买入但实际亏损）施以更高惩罚。

    金融逻辑：
      FP (y=0, pred≈1): 真实没有机会但买了 → 亏真金白银，代价高
      FN (y=1, pred≈0): 有机会但没买   → 只是踏空，代价低

    weight_i = fp_penalty if y_i=0 else fn_penalty
    grad_i   = weight_i × (p - y)
    """
    def __init__(self, fp_penalty: float = 2.5, fn_penalty: float = 1.0):
        self.fp_penalty = fp_penalty
        self.fn_penalty = fn_penalty

    def __call__(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y      = dtrain.get_label()
        p      = _sigmoid(y_pred)
        weight = np.where(y == 0, self.fp_penalty, self.fn_penalty)
        grad   = weight * (p - y)
        hess   = np.maximum(weight * p * (1 - p), 1e-6)
        return grad, hess


class DirectionalLossObjective:
    """
    方向准确率损失：对高置信度方向错误施加额外惩罚。

    当 p > 0.5 但 y = 0（自信地预测做多，实际应避免）或
       p < 0.5 但 y = 1（自信地预测回避，实际应做多）时，
    置信度越高，额外惩罚越大。

    confidence_i = |p_i - 0.5| × 2  ∈ [0, 1]
    extra_weight = 1 + direction_penalty × confidence  (仅方向错误时)
    """
    def __init__(self, direction_penalty: float = 2.0):
        self.direction_penalty = direction_penalty

    def __call__(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y          = dtrain.get_label()
        p          = _sigmoid(y_pred)
        wrong_dir  = ((p > 0.5) & (y == 0)) | ((p < 0.5) & (y == 1))
        confidence = np.abs(p - 0.5) * 2
        extra_w    = np.where(wrong_dir, 1 + self.direction_penalty * confidence, 1.0)
        grad       = extra_w * (p - y)
        hess       = np.maximum(extra_w * p * (1 - p), 1e-6)
        return grad, hess


class CombinedObjective:
    """
    可配置组合损失函数：Focal + Asymmetric + Directional 加权求和。

    grad_total = w_focal × grad_focal + w_asym × grad_asym + w_dir × grad_dir
    hess_total = w_focal × hess_focal + w_asym × hess_asym + w_dir × hess_dir

    通过 LossConfig 灵活调整各组件权重和超参数。
    """
    def __init__(self, loss_cfg: LossConfig):
        self.focal = FocalLossObjective(gamma=loss_cfg.focal_gamma)
        self.asym  = AsymmetricLossObjective(fp_penalty=loss_cfg.fp_penalty,
                                              fn_penalty=loss_cfg.fn_penalty)
        self.direc = DirectionalLossObjective(direction_penalty=loss_cfg.direction_penalty)
        self.w_f   = loss_cfg.focal_weight
        self.w_a   = loss_cfg.asymmetric_weight
        self.w_d   = loss_cfg.direction_weight

    def __call__(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        g_f, h_f = self.focal(y_pred, dtrain)
        g_a, h_a = self.asym (y_pred, dtrain)
        g_d, h_d = self.direc(y_pred, dtrain)
        grad = self.w_f * g_f + self.w_a * g_a + self.w_d * g_d
        hess = self.w_f * h_f + self.w_a * h_a + self.w_d * h_d
        return grad, np.maximum(hess, 1e-6)


def make_sharpe_eval(actual_rets: np.ndarray, threshold: float = 0.5,
                     periods_per_year: float = 50.0):
    """
    夏普比率评估指标（用于验证集早停）。

    原理：用模型输出的买入概率 p_i 作为信号权重，
         计算 pnl_i = signal_i × actual_ret_i，
         然后计算年化夏普比率。

    注意：actual_rets 必须与 DMatrix 行顺序完全对齐。
    periods_per_year: 每年约多少个调仓周期（默认 50 = 每 5 交易日调仓）。
    """
    rets = actual_rets.copy()

    def sharpe_metric(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        p = _sigmoid(y_pred)
        # 用软信号（概率）而非二值信号：grad 非零，避免训练初期 sharpe 恒 0 触发假早停
        # pnl_i = p_i × ret_i：高概率买入 → 获取实际持有期收益
        pnl = p * rets[:len(p)]
        std = pnl.std()
        if std < 1e-8:
            return 'sharpe', 0.0
        sharpe = pnl.mean() / std * np.sqrt(periods_per_year)
        return 'sharpe', float(sharpe)

    return sharpe_metric


# ══════════════════════════════════════════════════════════════════════════════
# 8. 模型训练
# ══════════════════════════════════════════════════════════════════════════════

def purged_cutoff(all_dates: list, cutoff: str, purge_n: int) -> str:
    """
    Purged Cross-Validation helper（Lopez de Prado 方法）。
    返回 cutoff 往前 purge_n 个调仓日的日期字符串，用于在训练集末尾
    清除标签与验证集重叠的样本，防止 Label Overlap 数据泄露。

    例如 max_hold=15，则截面日 t 的标签延伸至 t+15 交易日；
    val_cutoff 之前 15 个调仓日的训练样本其标签已跨入验证期，必须清除。
    """
    dates_before = [d for d in all_dates if d < cutoff]
    if len(dates_before) <= purge_n:
        return all_dates[0] if all_dates else cutoff
    return dates_before[-(purge_n + 1)]  # 倒数第 purge_n 个（含）之前


def split_panel(panel: pd.DataFrame, cfg: ModelConfig):
    """Purged Time-Series Split：训练集 / 验证集 / 测试集

    按 Lopez de Prado《金融机器学习》方法：在训练集末尾去掉
    max_hold 个调仓日的样本（其标签延伸入验证期），防止数据泄露。
    """
    panel['trade_date'] = panel['trade_date'].astype(str)
    all_dates = sorted(panel['trade_date'].unique())

    train = panel[(panel['trade_date'] >= cfg.train_start) &
                  (panel['trade_date'] <= cfg.train_end)].copy()
    val   = train[train['trade_date'] >= cfg.val_cutoff].copy()

    # Purging：清除训练集末尾标签与验证集重叠的部分
    purge_cut = purged_cutoff(all_dates, cfg.val_cutoff, purge_n=cfg.max_hold)
    tr = train[train['trade_date'] < purge_cut].copy()

    # Embargo：主测试集在 train_end 后已有 ~1个月隔离带（test_start="20230201"），无需额外处理
    test  = panel[panel['trade_date'] >= cfg.test_start].copy()

    print(f"  训练子集: {tr['trade_date'].min()} ~ {tr['trade_date'].max()}, {len(tr):,} 行")
    print(f"    (已 Purge {cfg.max_hold} 个调仓日，防止 Label Overlap)")
    print(f"  验证子集: {val['trade_date'].min()} ~ {val['trade_date'].max()}, {len(val):,} 行")
    print(f"  测试集  : {test['trade_date'].min()} ~ {test['trade_date'].max()}, {len(test):,} 行")
    return tr, val, test


def train_timing_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: List[str],
    cfg: ModelConfig,
) -> xgb.Booster:
    """
    训练 XGBoost 择时模型，使用自定义组合损失函数。
    早停指标：验证集夏普比率（最大化）。
    """
    print("\n[训练] XGBoost 时序择时模型...")
    X_tr  = train[feature_cols].fillna(0).values.astype(float)
    y_tr  = train['binary_label'].values.astype(float)
    X_val = val[feature_cols].fillna(0).values.astype(float)
    y_val = val['binary_label'].values.astype(float)

    # 正样本率诊断
    pos_rate_tr  = y_tr.mean()
    pos_rate_val = y_val.mean()
    print(f"  训练正样本率: {pos_rate_tr:.2%} ({int(y_tr.sum()):,}/{len(y_tr):,})")
    print(f"  验证正样本率: {pos_rate_val:.2%} ({int(y_val.sum()):,}/{len(y_val):,})")

    # 样本权重：对正样本上权 = neg/pos，缓解类不平衡（补充自定义损失的效果）
    # 使用平方根权，避免过度上权导致训练不稳定
    if pos_rate_tr > 0:
        pos_weight = np.sqrt((1 - pos_rate_tr) / pos_rate_tr)
        sample_weight = np.where(y_tr == 1, pos_weight, 1.0)
    else:
        sample_weight = np.ones_like(y_tr)

    dtrain = xgb.DMatrix(X_tr,  label=y_tr, weight=sample_weight)
    dval   = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth':        cfg.max_depth,
        'learning_rate':    cfg.learning_rate,
        'subsample':        cfg.subsample,
        'colsample_bytree': cfg.colsample_bytree,
        'min_child_weight': cfg.min_child_weight,
        'reg_lambda':       cfg.reg_lambda,
        'reg_alpha':        cfg.reg_alpha,
        'tree_method':      'hist',
        'seed':             42,
        'nthread':          -1,
        'disable_default_eval_metric': 1,
    }

    combined_obj = CombinedObjective(cfg.loss)

    # 用 logloss 作为早停指标（比 Sharpe 更稳定，不受单一验证期市场环境干扰）
    # Sharpe 在事后评测中单独报告
    def logloss_eval(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        p = np.clip(_sigmoid(y_pred), 1e-7, 1 - 1e-7)
        # 正样本上权（与训练权重一致）以免 logloss 被负样本主导
        w = np.where(y == 1, pos_weight if pos_rate_tr > 0 else 1.0, 1.0)
        logloss = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean()
        return 'logloss', float(logloss)

    print(f"  损失函数: Focal({cfg.loss.focal_weight:.2f}) + "
          f"Asymmetric({cfg.loss.asymmetric_weight:.2f}) + "
          f"Directional({cfg.loss.direction_weight:.2f})")
    print(f"  fp_penalty={cfg.loss.fp_penalty}, fn_penalty={cfg.loss.fn_penalty}, "
          f"gamma={cfg.loss.focal_gamma}")
    print(f"  早停指标: 加权 logloss（验证集），{cfg.early_stopping_rounds} 轮无改善停止")

    callbacks = [xgb.callback.EarlyStopping(
        rounds=cfg.early_stopping_rounds, maximize=False,  # minimize logloss
        metric_name='logloss', save_best=True)]

    model = xgb.train(
        params          = params,
        dtrain          = dtrain,
        num_boost_round = cfg.n_estimators,
        obj             = combined_obj,
        evals           = [(dval, 'val')],
        custom_metric   = logloss_eval,
        callbacks       = callbacks,
        verbose_eval    = 100,
    )
    best_iter = getattr(model, 'best_iteration', 0)
    print(f"  最优迭代: {best_iter} 轮")

    # 事后在验证集上报告 Sharpe 和最优阈值
    X_v  = val[feature_cols].fillna(0).values.astype(float)
    prob_val = predict_proba(model, X_v)
    # 用 5 日持有期收益（与调仓周期匹配），fallback 到 actual_ret
    ret_col_val = 'port_ret' if 'port_ret' in val.columns else 'actual_ret'
    val_rets = val[ret_col_val].fillna(0).values
    print("  验证集阈值扫描:")
    best_thresh, best_sharpe_val = cfg.signal_threshold, -np.inf
    for th in np.arange(0.10, 0.55, 0.05):
        sig  = (prob_val >= th).astype(float)
        n_sig = int(sig.sum())
        if n_sig == 0:
            continue
        pnl  = sig * val_rets
        std  = pnl.std()
        sh   = pnl.mean() / std * np.sqrt(50) if std > 1e-8 else 0.0
        prec = precision_score(y_val, (prob_val >= th).astype(int), zero_division=0)
        print(f"    th={th:.2f} | n_signals={n_sig:,} Precision={prec:.3f} Sharpe={sh:.3f}")
        if sh > best_sharpe_val:
            best_sharpe_val = sh
            best_thresh = float(th)
    print(f"  → 最优验证集阈值: {best_thresh:.2f}  (Sharpe={best_sharpe_val:.3f})")
    model._best_threshold = best_thresh   # 附加到 model 供后续使用
    return model


def predict_proba(model: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """预测买入概率（将 logit 转为概率）"""
    dmat = xgb.DMatrix(X)
    raw  = model.predict(dmat)
    return _sigmoid(raw)


# ══════════════════════════════════════════════════════════════════════════════
# 9. 评测
# ══════════════════════════════════════════════════════════════════════════════

def compute_rank_ic(panel: pd.DataFrame, label: str = '') -> dict:
    """
    计算截面 Rank IC（Spearman 秩相关系数）序列及 ICIR。

    在量化选股评测中，Rank IC 比 AUC 更具业务含义：
      - IC > 0.03 为可用信号，> 0.05 为强信号
      - ICIR = IC均值/IC标准差 > 0.5 为稳健信号（类似因子夏普比率）

    使用 port_ret（5日持有期）作为真实收益，pred_prob 作为预测排名依据。
    """
    from scipy.stats import spearmanr

    ret_col = 'port_ret' if 'port_ret' in panel.columns else 'actual_ret'
    ic_list = []
    for dt, grp in panel.groupby('trade_date'):
        valid = grp[['pred_prob', ret_col]].dropna()
        if len(valid) < 10:
            continue
        ic, _ = spearmanr(valid['pred_prob'], valid[ret_col])
        if not np.isnan(ic):
            ic_list.append(ic)

    if not ic_list:
        return {}

    ics = np.array(ic_list)
    ic_mean   = ics.mean()
    ic_std    = ics.std()
    icir      = ic_mean / ic_std if ic_std > 1e-8 else 0.0
    ic_pos    = (ics > 0).mean()

    tag = f"[{label}] " if label else ""
    print(f"\n  {tag}Rank IC / ICIR:")
    print(f"    IC 均值    : {ic_mean:+.4f}  (> 0.03 为可用信号)")
    print(f"    IC 标准差  : {ic_std:.4f}")
    print(f"    ICIR       : {icir:+.3f}   (> 0.5 为稳健信号)")
    print(f"    IC > 0 比例: {ic_pos:.1%}  ({len(ics)} 个截面)")

    return dict(ic_mean=ic_mean, ic_std=ic_std, icir=icir,
                ic_pos_rate=ic_pos, n_periods=len(ics))


def evaluate_ml_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                        threshold: float = 0.5, label: str = '') -> dict:
    """
    算法层面评测：
      Accuracy / Precision / Recall / F1 / AUC / 混淆矩阵
    """
    y_pred = (y_prob >= threshold).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    cm = confusion_matrix(y_true, y_pred)

    n  = len(y_true)
    n1 = int(y_true.sum())
    print(f"\n  [ML 评测 {label}] 样本={n:,}, 正样本={n1:,} ({n1/n:.1%}), "
          f"threshold={threshold:.2f}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}  (买入信号精确率：买了多少比例真的涨)")
    print(f"    Recall   : {rec:.4f}  (覆盖率：实际上涨中被识别的比例)")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    AUC-ROC  : {auc:.4f}")
    print(f"    混淆矩阵 :\n{cm}")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc,
                confusion_matrix=cm, threshold=threshold)


def simulate_portfolio(panel: pd.DataFrame, cfg: ModelConfig,
                       label: str = '', tranches: int = 1) -> dict:
    """
    金融回测，支持滚动分仓（Rolling Tranches）。

    tranches=1（默认）：单一组合，每个调仓日全量换仓。
    tranches=K（K>1）：将资金等分为 K 份，每份在不同调仓日轮转调仓。
      - 子组合 k 在 dates[i % K == k] 的日期调仓，其他日期继续持有；
      - 换手成本仅在调仓日产生；
      - 综合回报 = K 个子组合净收益的等权均值。
    作用：平滑"时序择时运气"（Timing Luck），减少因单一调仓日选择
    导致的净值曲线抖动，类似于 Gemini 对话中描述的 Rolling Tranches 机制。
    """
    tc_round = cfg.commission * 2 + cfg.stamp_tax + cfg.slippage * 2
    ret_col  = 'port_ret' if 'port_ret' in panel.columns else 'actual_ret'

    panel   = panel.sort_values('trade_date').copy()
    dates   = sorted(panel['trade_date'].unique())
    nav     = 1.0
    nav_log = []

    K = max(1, tranches)
    # 每个子组合维护当前持仓集合
    sub_holdings: List[set] = [set() for _ in range(K)]

    for i, dt in enumerate(dates):
        day_data = panel[panel['trade_date'] == dt].copy()

        period_net_ret = 0.0  # 本期加权净收益（K 个子组合均值）
        period_n_stocks = 0

        for k in range(K):
            if i % K == k:
                # 子组合 k 今日调仓：检查市场宽度，宽度不足时暂停新开仓
                breadth = (day_data['breadth_pct_ma20'].iloc[0]
                           if 'breadth_pct_ma20' in day_data.columns else 1.0)
                if breadth < cfg.min_breadth:
                    selected = []  # 市场弱势，持现金
                else:
                    candidates = day_data[day_data['pred_prob'] >= cfg.signal_threshold]
                    selected   = (candidates.nlargest(cfg.top_n, 'pred_prob')['ts_code']
                                  .tolist() if len(candidates) > 0 else [])
                new_set  = set(selected)
                old_set  = sub_holdings[k]
                turnover = (len(new_set.symmetric_difference(old_set))
                            / max(len(new_set | old_set), 1))
                cost = turnover * tc_round
                sub_holdings[k] = new_set
            else:
                # 继续持有，不产生交易成本
                cost = 0.0

            # 子组合本期收益
            held = sub_holdings[k]
            if held:
                sub_rets = day_data[day_data['ts_code'].isin(held)][ret_col].dropna()
                sub_ret  = sub_rets.mean() if len(sub_rets) > 0 else 0.0
            else:
                sub_ret = 0.0

            period_net_ret  += (sub_ret - cost) / K
            period_n_stocks += len(held)

        nav *= (1 + period_net_ret)
        nav_log.append({'trade_date': dt, 'nav': nav,
                        'net_ret': period_net_ret,
                        'n_stocks': period_n_stocks // K})

    if not nav_log:
        return {}

    nav_df = pd.DataFrame(nav_log).set_index('trade_date')
    rets   = nav_df['net_ret']   # 净收益（已含成本），与 NAV 完全一致

    # 年化收益（基于期末 NAV）
    periods_per_year = 252 / cfg.rebal_freq
    n_periods  = len(rets)
    annual_ret = nav_df['nav'].iloc[-1] ** (periods_per_year / n_periods) - 1

    # 最大回撤
    roll_max = nav_df['nav'].cummax()
    dd       = nav_df['nav'] / roll_max - 1
    max_dd   = dd.min()

    # 夏普（净收益，年化）
    sharpe = rets.mean() / rets.std() * np.sqrt(periods_per_year) if rets.std() > 1e-8 else 0.0

    # 卡玛
    calmar = annual_ret / abs(max_dd) if max_dd < 0 else np.nan

    # 胜率 / 盈亏比
    pos = rets[rets > 0]
    neg = rets[rets < 0]
    win_rate = (rets > 0).mean()
    pl_ratio = (pos.mean() / abs(neg.mean())) if (len(pos) > 0 and len(neg) > 0) else np.nan

    tranche_tag = f"  滚动分仓: {K} 仓" if K > 1 else ""
    print(f"\n  [金融回测 {label}]{tranche_tag}  期间: {nav_df.index[0]} ~ {nav_df.index[-1]}")
    print(f"    年化收益  : {annual_ret:+.2%}")
    print(f"    最大回撤  : {max_dd:.2%}")
    print(f"    夏普比率  : {sharpe:.3f}")
    print(f"    卡玛比率  : {calmar:.3f}" if not np.isnan(calmar) else "    卡玛比率  : N/A")
    print(f"    胜率      : {win_rate:.2%}")
    print(f"    盈亏比    : {pl_ratio:.2f}" if not np.isnan(pl_ratio) else "    盈亏比    : N/A")
    print(f"    持仓期数  : {n_periods}")

    return dict(annual_ret=annual_ret, max_dd=max_dd, sharpe=sharpe,
                calmar=calmar, win_rate=win_rate, pl_ratio=pl_ratio,
                nav_df=nav_df, label=label)


def walk_forward_evaluation(
    panel: pd.DataFrame,
    feature_cols: List[str],
    cfg: ModelConfig,
    tranches: int = 1,
) -> List[dict]:
    """
    滚动样本外测试（Walk-Forward Optimization）。

    训练窗口滚动扩展，每年测试一个窗口：
      [train_start, year-1年末] → 预测当年
    每个窗口独立训练模型，报告样本外绩效。
    """
    print("\n" + "=" * 60)
    print("  [鲁棒性] Walk-Forward Optimization (WFO)")
    print("=" * 60)

    test_years = [2020, 2021, 2022, 2023, 2024]
    results    = []

    for year in test_years:
        tr_end   = f"{year-1}1231"
        te_start = f"{year}0101"
        te_end   = f"{year}1231"

        train_data = panel[(panel['trade_date'] >= cfg.train_start) &
                           (panel['trade_date'] <= tr_end)].copy()
        test_data  = panel[(panel['trade_date'] >= te_start) &
                           (panel['trade_date'] <= te_end)].copy()

        if len(train_data) < 1000 or len(test_data) < 50:
            print(f"  {year}: 数据不足，跳过")
            continue

        # 拆分 val（训练集最后半年）并 Purge 标签重叠区间
        val_cut = f"{year-1}0601"
        val = train_data[train_data['trade_date'] >= val_cut].copy()
        all_tr_dates = sorted(panel['trade_date'].unique())
        purge_cut = purged_cutoff(all_tr_dates, val_cut, purge_n=cfg.max_hold)
        tr  = train_data[train_data['trade_date'] <  purge_cut].copy()

        if len(tr) < 500 or len(val) < 50:
            n = len(train_data)
            val = train_data.tail(int(n * 0.2)).copy()
            purge_cut2 = purged_cutoff(sorted(train_data['trade_date'].unique()),
                                       val['trade_date'].min(), purge_n=cfg.max_hold)
            tr  = train_data[train_data['trade_date'] < purge_cut2].copy()

        try:
            mdl = train_timing_model(tr, val, feature_cols, cfg)
        except Exception as e:
            print(f"  {year}: 训练失败 ({e})")
            continue

        X_te = test_data[feature_cols].fillna(0).values.astype(float)
        test_data = test_data.copy()
        test_data['pred_prob'] = predict_proba(mdl, X_te)

        # 使用验证集优化的最优阈值
        best_thresh = getattr(mdl, '_best_threshold', cfg.signal_threshold)

        # ML 指标
        ml = evaluate_ml_metrics(
            test_data['binary_label'].values,
            test_data['pred_prob'].values,
            best_thresh, label=str(year))

        # Rank IC（量化标准信号质量指标）
        ic = compute_rank_ic(test_data, label=str(year))

        # 金融指标（用最优阈值，K=3 滚动分仓）
        cfg_wfo = ModelConfig(**{k: v for k, v in cfg.__dict__.items() if k != 'loss'})
        cfg_wfo.signal_threshold = best_thresh
        cfg_wfo.loss = cfg.loss
        fin = simulate_portfolio(test_data, cfg_wfo, label=str(year), tranches=tranches)

        results.append({
            'year':        year,
            'n_train':     len(train_data),
            'n_test':      len(test_data),
            **{k: ml[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']},
            'ic_mean':     ic.get('ic_mean', np.nan),
            'icir':        ic.get('icir', np.nan),
            'annual_ret':  fin.get('annual_ret', np.nan),
            'max_dd':      fin.get('max_dd', np.nan),
            'sharpe':      fin.get('sharpe', np.nan),
            'win_rate':    fin.get('win_rate', np.nan),
        })

    if results:
        wfo_df = pd.DataFrame(results)
        print("\n  WFO 汇总:")
        print(wfo_df[['year', 'auc', 'ic_mean', 'icir',
                       'annual_ret', 'max_dd', 'sharpe']].to_string(index=False))
    return results


def sensitivity_analysis(
    panel: pd.DataFrame,
    feature_cols: List[str],
    cfg: ModelConfig,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    参数敏感性分析：测试关键参数变动对样本外 Sharpe 的影响。
    固定其他参数，每次只变动一个参数。
    """
    print("\n" + "=" * 60)
    print("  [鲁棒性] 参数敏感性分析")
    print("=" * 60)

    param_grid = {
        'profit_take':   [1.0, 1.5, 2.0, 2.5],
        'stop_loss':     [0.5, 1.0, 1.5],
        'fp_penalty':    [1.5, 2.0, 2.5, 3.0, 4.0],
        'focal_gamma':   [1.0, 2.0, 3.0],
        'signal_thresh': [0.4, 0.45, 0.5, 0.55, 0.6],
    }

    rows = []
    # 用已有的 test_data（有 pred_prob 列）测试 signal_threshold 敏感性
    for thresh in param_grid['signal_thresh']:
        cfg_tmp = ModelConfig(**{k: v for k, v in cfg.__dict__.items()
                                 if k != 'loss'})
        cfg_tmp.signal_threshold = thresh
        cfg_tmp.loss = cfg.loss
        test_copy = test_data.copy()
        fin = simulate_portfolio(test_copy, cfg_tmp, label=f'thresh={thresh}')
        rows.append({'param': 'signal_threshold', 'value': thresh,
                     'sharpe': fin.get('sharpe', np.nan),
                     'annual_ret': fin.get('annual_ret', np.nan),
                     'max_dd': fin.get('max_dd', np.nan)})

    # fp_penalty 敏感性（需重训练）
    val_cut = cfg.val_cutoff
    tr_data  = panel[(panel['trade_date'] >= cfg.train_start) &
                     (panel['trade_date'] <  val_cut)].copy()
    val_data = panel[(panel['trade_date'] >= val_cut) &
                     (panel['trade_date'] <= cfg.train_end)].copy()

    for fp in param_grid['fp_penalty']:
        loss_cfg = LossConfig(focal_weight=cfg.loss.focal_weight,
                              asymmetric_weight=cfg.loss.asymmetric_weight,
                              direction_weight=cfg.loss.direction_weight,
                              focal_gamma=cfg.loss.focal_gamma,
                              fp_penalty=fp,
                              fn_penalty=cfg.loss.fn_penalty)
        cfg_tmp = ModelConfig(loss=loss_cfg)
        try:
            mdl = train_timing_model(tr_data, val_data, feature_cols, cfg_tmp)
            X_te = test_data[feature_cols].fillna(0).values.astype(float)
            te_copy = test_data.copy()
            te_copy['pred_prob'] = predict_proba(mdl, X_te)
            fin = simulate_portfolio(te_copy, cfg_tmp, label=f'fp={fp}')
            rows.append({'param': 'fp_penalty', 'value': fp,
                         'sharpe': fin.get('sharpe', np.nan),
                         'annual_ret': fin.get('annual_ret', np.nan),
                         'max_dd': fin.get('max_dd', np.nan)})
        except Exception as e:
            print(f"  fp_penalty={fp} 训练失败: {e}")

    sens_df = pd.DataFrame(rows)
    print("\n  参数敏感性汇总:")
    print(sens_df.to_string(index=False))
    return sens_df


# ══════════════════════════════════════════════════════════════════════════════
# 10. 可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    test_data: pd.DataFrame,
    fin_result: dict,
    wfo_results: List[dict],
    sens_df: pd.DataFrame,
    model: xgb.Booster,
    feature_cols: List[str],
    cfg: ModelConfig,
):
    """生成综合评测图表（5 图）"""
    os.makedirs(f"{cfg.output_dir}/images", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/csv",    exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])   # 净值曲线
    ax2 = fig.add_subplot(gs[0, 2])    # 混淆矩阵
    ax3 = fig.add_subplot(gs[1, :])    # 每期信号概率分布
    ax4 = fig.add_subplot(gs[2, :2])   # 特征重要性
    ax5 = fig.add_subplot(gs[2, 2])    # WFO / 参数敏感性

    fig.suptitle("XGBoost 时序择时模型 — 综合评测", fontsize=14)

    # ── 1. 净值曲线 ──────────────────────────────────────────────────────────
    if 'nav_df' in fin_result:
        nav_df = fin_result['nav_df']
        ax1.plot(range(len(nav_df)), nav_df['nav'].values,
                 color='royalblue', linewidth=1.8, label='择时组合净值')
        ax1.axhline(1, color='black', linewidth=0.8, linestyle='--')
        # 标注最大回撤区间
        roll_max = nav_df['nav'].cummax()
        dd = nav_df['nav'] / roll_max - 1
        mdd_end   = dd.idxmin()
        mdd_val   = dd.min()
        idx_end   = list(nav_df.index).index(mdd_end)
        nav_sub   = nav_df['nav'][:mdd_end]
        mdd_start = nav_sub.idxmax()
        idx_start = list(nav_df.index).index(mdd_start)
        ax1.axvspan(idx_start, idx_end, alpha=0.15, color='red', label=f'MaxDD={mdd_val:.1%}')
        ax1.set_title(f"净值曲线 | 年化={fin_result.get('annual_ret',0):+.2%}  "
                      f"Sharpe={fin_result.get('sharpe',0):.2f}  "
                      f"MaxDD={mdd_val:.2%}")
        ax1.set_xlabel("调仓期序号")
        ax1.set_ylabel("净值")
        ax1.legend(fontsize=8)

    # ── 2. 混淆矩阵 ──────────────────────────────────────────────────────────
    cm = confusion_matrix(
        test_data['binary_label'].astype(int),
        (test_data['pred_prob'] >= cfg.signal_threshold).astype(int))
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['预测:避免', '预测:买入'])
    ax2.set_yticklabels(['实际:避免', '实际:买入'])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax2.set_title("混淆矩阵（测试集）")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # ── 3. 每期信号概率分布（买入信号率时序）──────────────────────────────────
    sig_rate = (test_data.groupby('trade_date')['pred_prob']
                .apply(lambda x: (x >= cfg.signal_threshold).mean()))
    ax3.bar(range(len(sig_rate)), sig_rate.values, color='steelblue', alpha=0.7)
    ax3.axhline(sig_rate.mean(), color='red', linewidth=1.2,
                linestyle='--', label=f'均值={sig_rate.mean():.1%}')
    ax3.set_title("每调仓日买入信号率（测试集）")
    ax3.set_xlabel("调仓期序号")
    ax3.set_ylabel("买入信号比例")
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=8)

    # ── 4. 特征重要性（Top 20）────────────────────────────────────────────────
    scores = model.get_score(importance_type='gain')
    imp    = pd.Series(scores).sort_values(ascending=True).tail(20)
    ax4.barh(range(len(imp)), imp.values, color='teal', alpha=0.8)
    ax4.set_yticks(range(len(imp)))
    ax4.set_yticklabels(imp.index, fontsize=7)
    ax4.set_title("特征重要性 (Gain, Top 20)")
    ax4.set_xlabel("Gain")

    # ── 5. WFO Sharpe 年度条形图 ──────────────────────────────────────────────
    if wfo_results:
        wfo_df = pd.DataFrame(wfo_results)
        colors = ['green' if s > 0 else 'red' for s in wfo_df['sharpe']]
        ax5.bar(wfo_df['year'].astype(str), wfo_df['sharpe'], color=colors, alpha=0.8)
        ax5.axhline(0, color='black', linewidth=0.8)
        ax5.axhline(wfo_df['sharpe'].mean(), color='blue', linewidth=1.2,
                    linestyle='--', label=f"均值={wfo_df['sharpe'].mean():.2f}")
        ax5.set_title("WFO 年度夏普比率（滚动样本外）")
        ax5.set_xlabel("年份")
        ax5.set_ylabel("Sharpe Ratio")
        ax5.legend(fontsize=8)

    plt.tight_layout()
    save_path = f"{cfg.output_dir}/images/xgb_market_timing_eval.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表已保存: {save_path}")

    # 保存特征重要性 CSV
    scores_full = model.get_score(importance_type='gain')
    feat_df = pd.DataFrame({'feature': list(scores_full.keys()),
                             'gain':   list(scores_full.values())})
    feat_df = feat_df.sort_values('gain', ascending=False)
    feat_df.to_csv(f"{cfg.output_dir}/csv/timing_feature_importance.csv", index=False)

    # 保存预测结果 CSV
    save_cols = ['ts_code', 'trade_date', 'pred_prob', 'binary_label', 'actual_ret', 'port_ret']
    test_data[[c for c in save_cols if c in test_data.columns]].to_csv(
        f"{cfg.output_dir}/csv/timing_test_predictions.csv", index=False)

    # 保存 WFO 结果
    if wfo_results:
        pd.DataFrame(wfo_results).to_csv(
            f"{cfg.output_dir}/csv/timing_wfo_results.csv", index=False)

    # 保存参数敏感性
    if not sens_df.empty:
        sens_df.to_csv(f"{cfg.output_dir}/csv/timing_sensitivity.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='XGBoost 时序择时模型')
    p.add_argument('--profit_take',       type=float, default=1.5)
    p.add_argument('--stop_loss',         type=float, default=1.0)
    p.add_argument('--max_hold',          type=int,   default=15)
    p.add_argument('--focal_weight',      type=float, default=0.4)
    p.add_argument('--asymmetric_weight', type=float, default=0.4)
    p.add_argument('--direction_weight',  type=float, default=0.2)
    p.add_argument('--focal_gamma',       type=float, default=2.0)
    p.add_argument('--fp_penalty',        type=float, default=2.5)
    p.add_argument('--fn_penalty',        type=float, default=1.0)
    p.add_argument('--top_n',             type=int,   default=30)
    p.add_argument('--signal_threshold',  type=float, default=0.5)
    p.add_argument('--min_breadth',       type=float, default=0.0,
                   help='市场宽度门槛（默认0=不过滤；>0时需breadth_pct_ma20高于此值才开仓）')
    p.add_argument('--tranches',          type=int,   default=3,
                   help='滚动分仓数量（默认3：每仓等效持有 rebal_freq×K 天，减少低IC时段的交易频率）')
    p.add_argument('--skip_wfo',          action='store_true', help='跳过 WFO（加速调试）')
    p.add_argument('--skip_sensitivity',  action='store_true', help='跳过敏感性分析')
    return p.parse_args()


def main():
    t0   = time.time()
    args = parse_args()

    # ── 配置 ──────────────────────────────────────────────────────────────────
    loss_cfg = LossConfig(
        focal_weight      = args.focal_weight,
        asymmetric_weight = args.asymmetric_weight,
        direction_weight  = args.direction_weight,
        focal_gamma       = args.focal_gamma,
        fp_penalty        = args.fp_penalty,
        fn_penalty        = args.fn_penalty,
    )
    cfg = ModelConfig(
        profit_take      = args.profit_take,
        stop_loss        = args.stop_loss,
        max_hold         = args.max_hold,
        top_n            = args.top_n,
        signal_threshold = args.signal_threshold,
        min_breadth      = args.min_breadth,
        loss             = loss_cfg,
    )

    os.makedirs(f"{cfg.output_dir}/images", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/csv",    exist_ok=True)

    print("=" * 65)
    print("  XGBoost 时序择时模型  |  三重屏障法标签")
    print("=" * 65)
    hold_vol_factor = np.sqrt(cfg.max_hold / 252.0)
    import numpy as _np
    print(f"  上屏障 = entry × (1 + {cfg.profit_take} × σ_hold)  ≈ +{cfg.profit_take*0.35*hold_vol_factor:.1%} (σ=35%)")
    print(f"  下屏障 = entry × (1 - {cfg.stop_loss} × σ_hold)   ≈ -{cfg.stop_loss*0.35*hold_vol_factor:.1%} (σ=35%)")
    print(f"  最大持有: {cfg.max_hold} 交易日  持有期波动率因子: {hold_vol_factor:.4f}")

    conn = get_conn(cfg)
    try:
        # Step 1: 基础数据
        print("\n[Step 1] 价格 & 波动率矩阵")
        close_pivot, vol_pivot = load_price_matrices(conn, cfg)

        print("\n[Step 2] 调仓日期")
        rebal_dates = get_rebal_dates(conn, cfg)

        # Step 3: 三重屏障标签
        print("\n[Step 3] 三重屏障标签")
        label_pivot, ret_pivot = compute_triple_barrier_labels(close_pivot, vol_pivot, cfg)

        # 5日持有期收益（与调仓周期对齐，用于组合模拟）
        # close_pivot.shift(-rebal_freq) 是 rebal_freq 个交易日后的价格
        fwd_ret_pivot = close_pivot.shift(-cfg.rebal_freq) / close_pivot - 1
        print(f"  {cfg.rebal_freq}日持有期收益矩阵: {fwd_ret_pivot.shape}, "
              f"非空率 {fwd_ret_pivot.notna().mean().mean():.1%}")

        # Step 4: 技术/估值特征
        print("\n[Step 4] 技术 & 估值特征")
        tech_df = load_tech_val_features(conn, cfg)

        # Step 5: RSI
        print("\n[Step 5] RSI(14)")
        rsi_matrix = compute_rsi(close_pivot)

        # Step 6: 资金流向
        print("\n[Step 6] 资金流向特征")
        mf_df = load_moneyflow_features(conn, cfg)

        # Step 7: 基本面 PIT
        print("\n[Step 7] 基本面 PIT")
        fund_panel = load_fundamental_panel(conn)
        rebal_set  = set(rebal_dates)
        tech_rebal = tech_df[tech_df['trade_date'].isin(rebal_set)]
        rebal_keys = tech_rebal[['ts_code', 'trade_date']].drop_duplicates()
        fund_pit   = join_fundamental_pit(fund_panel, rebal_keys)

        # Step 8: 市场宽度
        print("\n[Step 8] 市场宽度特征")
        breadth_df = compute_market_breadth(close_pivot)

        # Step 9: 宏观特征
        print("\n[Step 9] 宏观 & 北向特征")
        macro_df = load_macro_features(conn, cfg)

        # Step 10: 股票基础信息
        stock_info = load_stock_info(conn)

    finally:
        conn.close()

    # Step 11: 组装 Panel
    print("\n[Step 11] 组装全量 Panel")
    panel = build_timing_panel(
        tech_df        = tech_df,
        rsi_matrix     = rsi_matrix[rsi_matrix.index.isin(rebal_set)],
        mf_df          = mf_df,
        fund_pit       = fund_pit,
        breadth_df     = breadth_df,
        macro_df       = macro_df,
        label_pivot    = label_pivot,
        ret_pivot      = ret_pivot,
        port_ret_pivot = fwd_ret_pivot,
        stock_info     = stock_info,
        rebal_dates    = rebal_dates,
        cfg            = cfg,
    )

    # Step 12: 特征非空率检查
    print("\n  各特征非空率:")
    avail_features = [c for c in ALL_FEATURES if c in panel.columns]
    for col in avail_features:
        rate = panel[col].notna().mean()
        flag = "⚠" if rate < 0.5 else "✓"
        print(f"    {flag} {col:25s}: {rate:.1%}")

    # Step 13: 预处理
    print("\n[Step 13] 数据预处理")
    panel = preprocess_panel(panel, avail_features, neutralize=True)

    # 过滤特征缺失过多的行
    na_rate = panel[avail_features].isna().mean(axis=1)
    panel   = panel[na_rate < 0.33].copy()

    # 剩余 NaN 用截面中位数填充
    for col in avail_features:
        if panel[col].isna().any():
            panel[col] = panel.groupby('trade_date')[col].transform(
                lambda x: x.fillna(x.median()))

    print(f"  最终 Panel: {len(panel):,} 行, {panel['trade_date'].nunique()} 截面")

    # Step 14: 训练/测试切分
    print("\n[Step 14] 训练/验证/测试切分")
    train_sub, val_sub, test_sub = split_panel(panel, cfg)

    # Step 15: 训练
    print("\n[Step 15] 模型训练")
    model = train_timing_model(train_sub, val_sub, avail_features, cfg)

    # Step 16: 测试集预测
    print("\n[Step 16] 测试集评测")
    X_test = test_sub[avail_features].fillna(0).values.astype(float)
    test_sub = test_sub.copy()
    test_sub['pred_prob'] = predict_proba(model, X_test)

    # 使用验证集上找到的最优阈值（如果模型附带了，否则用默认）
    best_thresh = getattr(model, '_best_threshold', cfg.signal_threshold)
    print(f"  使用阈值: {best_thresh:.2f} (来自验证集优化)")

    # 算法指标（测试集）
    ml_result = evaluate_ml_metrics(
        test_sub['binary_label'].values,
        test_sub['pred_prob'].values,
        best_thresh, label='TEST')

    # 阈值扫描（找最优 Precision/Recall 均衡点）
    print("\n  测试集阈值 vs Precision/Recall:")
    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        y_p = (test_sub['pred_prob'] >= th).astype(int)
        pr  = precision_score(test_sub['binary_label'], y_p, zero_division=0)
        rc  = recall_score(test_sub['binary_label'], y_p, zero_division=0)
        n   = y_p.sum()
        print(f"    threshold={th:.2f} | Precision={pr:.3f} Recall={rc:.3f} | 买入信号={n:,}")

    # Rank IC / ICIR（测试集，量化标准信号质量指标）
    ic_result = compute_rank_ic(test_sub, label='TEST')

    # 金融回测（测试集，使用最优阈值，K=3 滚动分仓）
    cfg_test = ModelConfig(**{k: v for k, v in cfg.__dict__.items() if k != 'loss'})
    cfg_test.signal_threshold = best_thresh
    cfg_test.loss = cfg.loss
    fin_result = simulate_portfolio(test_sub, cfg_test, label='TEST',
                                    tranches=args.tranches)

    # Step 17: 训练集评测（过拟合检验）
    print("\n[Step 17] 训练集评测（对比过拟合程度）")
    X_tr  = train_sub[avail_features].fillna(0).values.astype(float)
    train_sub = train_sub.copy()
    train_sub['pred_prob'] = predict_proba(model, X_tr)
    evaluate_ml_metrics(
        train_sub['binary_label'].values,
        train_sub['pred_prob'].values,
        best_thresh, label='TRAIN')

    # Step 18: WFO
    wfo_results = []
    if not args.skip_wfo:
        wfo_results = walk_forward_evaluation(panel, avail_features, cfg, tranches=args.tranches)

    # Step 19: 参数敏感性
    sens_df = pd.DataFrame()
    if not args.skip_sensitivity:
        sens_df = sensitivity_analysis(panel, avail_features, cfg, test_sub)

    # Step 20: 可视化 & 保存
    print("\n[Step 20] 可视化 & 保存")
    plot_results(test_sub, fin_result, wfo_results, sens_df, model, avail_features, cfg)

    # 总结
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  完成! 总耗时: {elapsed:.1f}s")
    print(f"  测试集 AUC={ml_result['auc']:.4f}  "
          f"Precision={ml_result['precision']:.4f}  F1={ml_result['f1']:.4f}")
    if ic_result:
        print(f"  Rank IC={ic_result['ic_mean']:+.4f}  "
              f"ICIR={ic_result['icir']:+.3f}  "
              f"IC>0={ic_result['ic_pos_rate']:.0%}")
    print(f"  组合 年化={fin_result.get('annual_ret',0):+.2%}  "
          f"夏普={fin_result.get('sharpe',0):.3f}  "
          f"MaxDD={fin_result.get('max_dd',0):.2%}  "
          f"(滚动{args.tranches}仓)")
    if wfo_results:
        wfo_sharpes = [r['sharpe'] for r in wfo_results if not np.isnan(r['sharpe'])]
        wfo_ics = [r.get('ic_mean', np.nan) for r in wfo_results]
        valid_ics = [v for v in wfo_ics if not np.isnan(v)]
        print(f"  WFO 夏普: 均值={np.mean(wfo_sharpes):.3f}  "
              f"一致性={np.mean([s > 0 for s in wfo_sharpes]):.0%} 年份为正")
        if valid_ics:
            print(f"  WFO Rank IC: 均值={np.mean(valid_ics):+.4f}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
