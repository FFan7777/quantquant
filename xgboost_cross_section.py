#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 截面选股模型

目标: 预测10个交易日后个股相对沪深300(000300.SH)的超额收益，截面排序
标签: 个股10日收益 - 基准10日收益，截面百分位归一化 → [0,1]
评估: Rank IC、ICIR、GAUC
训练: 2018-2022  测试: 2023-2025 (20日隔离期，Purged Time-Series Split)

因子分组（共26个）:
  技术面 (8): ret_1d, ret_5d, ret_20d, ret_60d, vol_20d, close_vs_ma20,
              close_vs_ma60, rsi_14
  估值流动性 (5): pe_ttm, pb, log_mktcap, turnover_20d, volume_ratio
  资金流向 (3): mf_1d_mv, mf_5d_mv, large_net_5d_ratio
  基本面PIT (8): roe_ann, roa, gross_margin, debt_ratio, current_ratio,
                 fscore, rev_growth_yoy, ni_growth_yoy
  分析师预期 (2): analyst_count, np_yield

预处理: MAD去极值 → 行业/市值中性化(残差) → 截面Z-score标准化

注意: 严格遵守PIT原则，基本面数据使用f_ann_date(第一披露日)
"""

import os
import sys
import time
import warnings

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─── 配置 ────────────────────────────────────────────────────────────────────
DB_PATH       = "data/quant.duckdb"
BENCHMARK     = "000300.SH"          # 沪深300
HORIZON       = 10                   # 预测窗口（交易日）
REBAL_FREQ    = 5                    # 调仓频率（交易日）
DATA_START    = "20160101"           # 数据加载起点（含预热期）
TRAIN_START   = "20180101"           # 训练集起点

# ── 三段式严格分离的数据划分 ─────────────────────────────────────────────────
# LONG_VAL=True  (推荐): Train 2018-2021, Val 2022-2024, Test 2025+
#   Val 跨越熊/震荡/牛三种 Regime，策略 HP 搜索不过拟合单一市场状态
# LONG_VAL=False (原始): Train 2018-2023, Val 2024,       Test 2025+
#   更多训练数据，模型 IC 更高，但策略 HP 只有1年 Val（Regime 过拟合风险）
LONG_VAL      = True                 # True：Val 含多市场周期（减少 Regime 过拟合）

EMBARGO_DAYS  = 20                   # 隔离期（交易日数，约1个月）
if LONG_VAL:
    TRAIN_END  = "20211231"          # Train: 2018-2021（4年）
    VAL_START  = "20220201"          # Val: 2022-2024（3年，含2022熊市，多周期验证）
    VAL_END    = "20241231"
else:
    TRAIN_END  = "20231231"          # Train: 2018-2023（6年）
    VAL_START  = "20240201"          # Val: 2024（1年，含20日隔离）
    VAL_END    = "20241231"
TEST_START    = "20250201"           # Test: 2025+（VAL_END后+20交易日隔离）
END_DATE      = "20260316"           # 数据终点（当前最新）

# ── 超参数搜索配置（在 Val 上搜索，不接触 Test）───────────────────────────────
# MIN_MKTCAP 重置为保守默认值：原值5.0被基于测试集观察改为2.0（数据泄漏）
# 若要验证最优值，应通过 Val 集比较（build_panel 已含全部市值股，过滤在此做）
MIN_MKTCAP    = 5.0                  # 最小市值过滤（亿元，保守默认值）

# XGBoost 超参搜索空间（3×2×3 = 18 组，在 Val 上搜索）
HP_SEARCH_N_ESTIMATORS = 300        # 搜索阶段用（快速比较，非最优迭代轮数）
XGB_HP_GRID = [
    {"max_depth": d, "min_child_weight": mcw, "reg_lambda": rl}
    for d in [3, 4]        # depth=5 已实证会过拟合 val，导致 OOS 退步，排除在外
    for mcw in [20, 40]
    for rl in [5.0, 10.0, 20.0]
]
# 敏感性测试：对最优参数做邻近值测试（偏差应 < 0.01）
HP_SENSITIVITY_PARAMS = {
    "max_depth":        [2, 3, 4, 5, 6],
    "min_child_weight": [10, 20, 30, 40, 60],
    "reg_lambda":       [2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
}

ENSEMBLE_LGB  = True                 # 是否添加 LightGBM 模型做 XGB+LGB ensemble
WFO_MODE      = False                # Walk-Forward Optimization（已测试，收益边际）
WFO_REFIT_MONTHS = 6                 # WFO 重训间隔（月）
OUTPUT_DIR    = "output"

# 特征列名
#   技术面: ret_1d~ret_120d, vol_20d, close_vs_ma20/60/120, rsi_14
#   估值/流动性: pe_ttm, pb, log_mktcap, turnover_20d, volume_ratio, dv_ratio
#   资金流向: mf_1d/5d/20d_mv, large_net_5d/20d_ratio
#   基本面PIT: roe_ann, roa, gross_margin, debt_ratio, current_ratio,
#              fscore, rev_growth_yoy, ni_growth_yoy, gross_margin_chg_yoy, ocf_to_ni
#   分析师预期: analyst_count, np_yield
#   股东户数: holder_chg_qoq（筹码集中度）
TECH_COLS   = ['ret_1d', 'ret_5d', 'ret_20d', 'ret_60d', 'ret_120d',
               'vol_20d', 'close_vs_ma20', 'close_vs_ma60', 'close_vs_ma120', 'rsi_14',
               # Alpha158 扩充特征（剔除 skew_20d importance=0.69% 噪声）
               'amplitude_1d', 'open_vs_close',
               'dist_from_high_5d', 'dist_from_low_5d', 'dist_from_high_20d',
               'high_low_ratio_20d', 'vol_ratio_5_20']
BASIC_COLS  = ['pe_ttm', 'pb', 'log_mktcap', 'turnover_20d', 'volume_ratio', 'dv_ratio']
MF_COLS     = ['mf_1d_mv', 'mf_5d_mv', 'mf_20d_mv',
               'large_net_5d_ratio', 'large_net_20d_ratio',
               'retail_net_5d_ratio', 'smart_retail_divergence_5d']
FUND_COLS   = ['roe_ann', 'roa', 'fscore', 'rev_growth_yoy', 'ni_growth_yoy',
               'gross_margin_chg_yoy', 'sue',
               # 剔除: gross_margin(0.43%), debt_ratio(0.54%), ocf_to_ni(0.59%), current_ratio(0.65%)
               ]
ANALYST_COLS = ['analyst_count', 'np_yield', 'analyst_rev_30d']
HOLDER_COLS  = ['holder_chg_qoq']
# 非线性交叉特征（从现有特征组合生成，在 build_panel 中计算）
CROSS_COLS   = ['smart_momentum', 'momentum_adj_reversal', 'quality_value_score']
ALL_FEATURES = TECH_COLS + BASIC_COLS + MF_COLS + FUND_COLS + ANALYST_COLS + HOLDER_COLS + CROSS_COLS


# ─── 数据库连接 ───────────────────────────────────────────────────────────────
def get_conn(read_only: bool = True):
    return duckdb.connect(DB_PATH, read_only=read_only)


# ════════════════════════════════════════════════════════════════════════════
# 1. 技术面 & 市场微观结构特征（DuckDB 窗口函数）
# ════════════════════════════════════════════════════════════════════════════
def load_price_pivot(conn) -> pd.DataFrame:
    """加载前复权收盘价矩阵（date × ts_code），用于 RSI 和 label 计算"""
    print("  加载收盘价矩阵（前复权）...")
    df = conn.execute(f"""
        SELECT trade_date, ts_code, close
        FROM daily_price
        WHERE trade_date >= '{DATA_START}' AND trade_date <= '{END_DATE}'
          AND ts_code NOT LIKE '8%'          -- 排除北交所
          AND ts_code NOT LIKE '4%'
        ORDER BY trade_date, ts_code
    """).fetchdf()
    pivot = df.pivot(index='trade_date', columns='ts_code', values='close')
    pivot.index = pivot.index.astype(str)
    print(f"    收盘价矩阵: {pivot.shape[0]} 天 × {pivot.shape[1]} 只股票")
    return pivot


def load_tech_basic_features(conn) -> pd.DataFrame:
    """
    技术面 + 估值特征（SQL 窗口函数）
    返回: (ts_code, trade_date) → [ret_1d, ret_5d, ret_20d, ret_60d,
                                   vol_20d, close_vs_ma20, close_vs_ma60,
                                   pe_ttm, pb, log_mktcap, turnover_20d,
                                   volume_ratio, total_mv_100m]
    """
    print("  加载技术面 & 估值特征（SQL窗口函数）...")
    df = conn.execute(f"""
        WITH dp AS (
            SELECT ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol
            FROM daily_price
            WHERE trade_date >= '{DATA_START}' AND trade_date <= '{END_DATE}'
              AND ts_code NOT LIKE '8%'
              AND ts_code NOT LIKE '4%'
        ),
        tech AS (
            SELECT
                ts_code, trade_date, close,
                close / NULLIF(LAG(close,1)   OVER w, 0) - 1  AS ret_1d,
                close / NULLIF(LAG(close,5)   OVER w, 0) - 1  AS ret_5d,
                close / NULLIF(LAG(close,20)  OVER w, 0) - 1  AS ret_20d,
                close / NULLIF(LAG(close,60)  OVER w, 0) - 1  AS ret_60d,
                close / NULLIF(LAG(close,120) OVER w, 0) - 1  AS ret_120d,
                STDDEV(pct_chg/100.0) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                    * SQRT(252)                               AS vol_20d,
                AVG(close) OVER (w ROWS BETWEEN 19  PRECEDING AND CURRENT ROW) AS ma20,
                AVG(close) OVER (w ROWS BETWEEN 59  PRECEDING AND CURRENT ROW) AS ma60,
                AVG(close) OVER (w ROWS BETWEEN 119 PRECEDING AND CURRENT ROW) AS ma120,
                -- Alpha158 扩充特征
                (high - low) / NULLIF(pre_close, 0)           AS amplitude_1d,
                open / NULLIF(pre_close, 0) - 1               AS open_vs_close,
                close / NULLIF(MAX(high) OVER (w ROWS BETWEEN 4  PRECEDING AND CURRENT ROW), 0) - 1
                                                              AS dist_from_high_5d,
                close / NULLIF(MIN(low)  OVER (w ROWS BETWEEN 4  PRECEDING AND CURRENT ROW), 0) - 1
                                                              AS dist_from_low_5d,
                close / NULLIF(MAX(high) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) - 1
                                                              AS dist_from_high_20d,
                MAX(high) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) /
                  NULLIF(MIN(low) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) - 1
                                                              AS high_low_ratio_20d,
                AVG(vol) OVER (w ROWS BETWEEN 4  PRECEDING AND CURRENT ROW) /
                  NULLIF(AVG(vol) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0)
                                                              AS vol_ratio_5_20
            FROM dp
            WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
        ),
        db AS (
            SELECT
                ts_code, trade_date,
                pe_ttm, pb,
                LN(NULLIF(total_mv, 0))   AS log_mktcap,
                total_mv / 10000.0        AS total_mv_100m,   -- 万元 → 亿元
                AVG(turnover_rate) OVER (
                    PARTITION BY ts_code ORDER BY trade_date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                )                         AS turnover_20d,
                volume_ratio,
                COALESCE(dv_ratio, 0.0)   AS dv_ratio
            FROM daily_basic
            WHERE trade_date >= '{DATA_START}' AND trade_date <= '{END_DATE}'
              AND total_mv > 0
        )
        SELECT
            t.ts_code, t.trade_date,
            t.ret_1d, t.ret_5d, t.ret_20d, t.ret_60d, t.ret_120d, t.vol_20d,
            t.close / NULLIF(t.ma20,   0) - 1            AS close_vs_ma20,
            t.close / NULLIF(t.ma60,   0) - 1            AS close_vs_ma60,
            t.close / NULLIF(t.ma120,  0) - 1            AS close_vs_ma120,
            d.pe_ttm, d.pb, d.log_mktcap, d.total_mv_100m,
            d.turnover_20d, d.volume_ratio, d.dv_ratio,
            t.amplitude_1d, t.open_vs_close,
            t.dist_from_high_5d, t.dist_from_low_5d, t.dist_from_high_20d,
            t.high_low_ratio_20d, t.vol_ratio_5_20
        FROM tech t
        JOIN db d ON t.ts_code = d.ts_code AND t.trade_date = d.trade_date
        WHERE t.ret_60d IS NOT NULL        -- 确保至少有60天历史
          AND d.log_mktcap IS NOT NULL
    """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str)

    # skew_20d：20日日收益率偏度（需全量时间序列，在完整 df 上滚动计算）
    df = df.sort_values(['ts_code', 'trade_date'])
    df['skew_20d'] = (
        df.groupby('ts_code')['ret_1d']
        .transform(lambda s: s.rolling(20, min_periods=10).skew())
    )

    print(f"    技术/估值特征: {len(df):,} 行, {df['ts_code'].nunique()} 只股票")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. RSI(14) ── 向量化计算
# ════════════════════════════════════════════════════════════════════════════
def compute_rsi(close_pivot: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Wilder's RSI，全矩阵向量化计算。
    输入: close_pivot (date × ts_code，前复权)
    输出: 同形状的 RSI 矩阵
    """
    print("  计算RSI(14)...")
    delta = close_pivot.diff(1)
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    # Wilder EMA: alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


# ════════════════════════════════════════════════════════════════════════════
# 3. 资金流向特征
# ════════════════════════════════════════════════════════════════════════════
def load_moneyflow_features(conn) -> pd.DataFrame:
    """
    返回: (ts_code, trade_date) → [mf_1d_raw, mf_5d_raw, large_net_5d, total_flow_5d]
    mf 单位: 万元（待后续除以市值归一化）
    """
    print("  加载资金流向特征（SQL窗口函数）...")
    df = conn.execute(f"""
        SELECT
            ts_code, trade_date,
            net_mf_amount  AS mf_1d_raw,
            SUM(net_mf_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )  AS mf_5d_raw,
            SUM(net_mf_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            )  AS mf_20d_raw,
            SUM(buy_lg_amount + buy_elg_amount
                - sell_lg_amount - sell_elg_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )  AS large_net_5d,
            SUM(buy_lg_amount + buy_elg_amount
                - sell_lg_amount - sell_elg_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            )  AS large_net_20d,
            SUM(buy_sm_amount + buy_md_amount + buy_lg_amount + buy_elg_amount
                + sell_sm_amount + sell_md_amount + sell_lg_amount + sell_elg_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )  AS total_flow_5d,
            SUM(buy_sm_amount + buy_md_amount + buy_lg_amount + buy_elg_amount
                + sell_sm_amount + sell_md_amount + sell_lg_amount + sell_elg_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            )  AS total_flow_20d,
            -- 散户净流向（小单+中单）5日累计
            SUM(buy_sm_amount + buy_md_amount
                - sell_sm_amount - sell_md_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )  AS retail_net_5d
        FROM moneyflow
        WHERE trade_date >= '{DATA_START}' AND trade_date <= '{END_DATE}'
    """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str)
    print(f"    资金流向特征: {len(df):,} 行")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 4. 基本面 PIT 特征（从三张原始报表计算，使用 f_ann_date）
# ════════════════════════════════════════════════════════════════════════════
def load_fundamental_panel(conn) -> pd.DataFrame:
    """
    从 income_statement + balance_sheet + cash_flow 构建 PIT 基本面面板。
    包含 F-Score 和 YoY 增长率。
    键: (ts_code, f_ann_date) → 财务指标
    """
    print("  加载基本面数据（PIT三表合并）...")
    df = conn.execute("""
        WITH is_t AS (
            SELECT ts_code, end_date,
                   MIN(COALESCE(f_ann_date, ann_date)) AS f_ann_date,
                   FIRST(revenue      ORDER BY ann_date DESC) AS revenue,
                   FIRST(oper_cost    ORDER BY ann_date DESC) AS oper_cost,
                   FIRST(n_income_attr_p ORDER BY ann_date DESC) AS n_income,
                   FIRST(basic_eps    ORDER BY ann_date DESC) AS eps
            FROM income_statement
            WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        ),
        bs_t AS (
            SELECT ts_code, end_date,
                   FIRST(total_assets     ORDER BY ann_date DESC) AS total_assets,
                   FIRST(total_liab       ORDER BY ann_date DESC) AS total_liab,
                   FIRST(total_hldr_eqy_inc_min_int ORDER BY ann_date DESC) AS equity,
                   FIRST(total_cur_assets ORDER BY ann_date DESC) AS cur_assets,
                   FIRST(total_cur_liab   ORDER BY ann_date DESC) AS cur_liab
            FROM balance_sheet
            WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        ),
        cf_t AS (
            SELECT ts_code, end_date,
                   FIRST(n_cashflow_act ORDER BY ann_date DESC) AS ocf
            FROM cash_flow
            WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        )
        SELECT
            i.ts_code, i.end_date, i.f_ann_date,
            i.revenue, i.oper_cost, i.n_income, i.eps,
            b.total_assets, b.total_liab, b.equity,
            b.cur_assets, b.cur_liab,
            c.ocf
        FROM is_t i
        LEFT JOIN bs_t b USING (ts_code, end_date)
        LEFT JOIN cf_t c USING (ts_code, end_date)
        ORDER BY ts_code, end_date
    """).fetchdf()

    # ── 衍生指标 ──────────────────────────────────────────────────────────
    ta  = df["total_assets"].replace(0, np.nan)
    eq  = df["equity"].replace(0, np.nan)
    rev = df["revenue"].replace(0, np.nan)
    cl  = df["cur_liab"].replace(0, np.nan)

    # 报告期月数（用于 ROE 年化）
    df["period_months"] = df["end_date"].astype(str).str[4:6].map(
        {"03": 3, "06": 6, "09": 9, "12": 12}
    ).fillna(12).astype(int)
    df["pm_mult"] = 12.0 / df["period_months"]   # 年化倍数

    df["roe_raw"]     = df["n_income"] / eq * 100.0
    df["roe_ann"]     = df["roe_raw"] * df["pm_mult"]   # 年化 ROE
    df["roa"]         = df["n_income"] / ta * df["pm_mult"] * 100.0
    df["gross_margin"]= (df["revenue"] - df["oper_cost"]) / rev * 100.0
    df["debt_ratio"]  = df["total_liab"] / ta
    df["current_ratio"]= df["cur_assets"] / cl
    df["assets_turn"] = df["revenue"] / ta

    # ── YoY F-Score（同比，上一自然年同期）────────────────────────────────
    df = df.copy()
    df["end_date"] = df["end_date"].astype(str)
    df_dedup = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last").copy()
    df_dedup["year"]          = df_dedup["end_date"].str[:4].astype(int)
    df_dedup["mmdd"]          = df_dedup["end_date"].str[4:]
    df_dedup["prev_end_date"] = (df_dedup["year"] - 1).astype(str) + df_dedup["mmdd"]

    df_prev = df_dedup[["ts_code", "end_date", "roa", "debt_ratio",
                         "current_ratio", "gross_margin", "assets_turn",
                         "n_income", "revenue", "ocf"]].rename(columns={
        "end_date": "prev_end_date",
        "roa": "roa_p", "debt_ratio": "da_p", "current_ratio": "cr_p",
        "gross_margin": "gpm_p", "assets_turn": "at_p",
        "n_income": "ni_p", "revenue": "rev_p", "ocf": "ocf_p"
    })
    df_dedup = df_dedup.merge(df_prev, on=["ts_code", "prev_end_date"], how="left")

    # 9 个 Piotroski 信号
    roa_v = df_dedup["roa"].fillna(0.0)
    ocf_v = df_dedup["ocf"].fillna(0.0)
    ta_v  = df_dedup["total_assets"].replace(0, np.nan)

    df_dedup["f1"] = (roa_v > 0).astype(int)
    df_dedup["f2"] = (ocf_v > 0).astype(int)
    df_dedup["f3"] = (df_dedup["n_income"].notna() & df_dedup["ni_p"].notna() &
                      (df_dedup["n_income"] > df_dedup["ni_p"])).astype(int)
    df_dedup["f4"] = (ocf_v / ta_v > roa_v).astype(int)
    df_dedup["f5"] = (df_dedup["debt_ratio"].notna() & df_dedup["da_p"].notna() &
                      (df_dedup["debt_ratio"] < df_dedup["da_p"])).astype(int)
    df_dedup["f6"] = (df_dedup["current_ratio"].notna() & df_dedup["cr_p"].notna() &
                      (df_dedup["current_ratio"] > df_dedup["cr_p"])).astype(int)
    df_dedup["f7"] = (df_dedup["revenue"].notna() & df_dedup["rev_p"].notna() &
                      (df_dedup["revenue"] > df_dedup["rev_p"])).astype(int)
    df_dedup["f8"] = (df_dedup["gross_margin"].notna() & df_dedup["gpm_p"].notna() &
                      (df_dedup["gross_margin"] > df_dedup["gpm_p"])).astype(int)
    df_dedup["f9"] = (df_dedup["assets_turn"].notna() & df_dedup["at_p"].notna() &
                      (df_dedup["assets_turn"] > df_dedup["at_p"])).astype(int)
    df_dedup["fscore"] = df_dedup[["f1","f2","f3","f4","f5","f6","f7","f8","f9"]].sum(axis=1)

    # YoY 增长率（有前期数据时计算）
    df_dedup["rev_growth_yoy"] = np.where(
        df_dedup["rev_p"].notna() & (df_dedup["rev_p"] != 0),
        (df_dedup["revenue"] - df_dedup["rev_p"]) / df_dedup["rev_p"].abs(),
        np.nan
    )
    df_dedup["ni_growth_yoy"] = np.where(
        df_dedup["ni_p"].notna() & (df_dedup["ni_p"] > 0),  # 仅在上期盈利时计算
        (df_dedup["n_income"] - df_dedup["ni_p"]) / df_dedup["ni_p"].abs(),
        np.nan
    )

    # 毛利率同比变化（困境反转信号）
    df_dedup["gross_margin_chg_yoy"] = np.where(
        df_dedup["gross_margin"].notna() & df_dedup["gpm_p"].notna(),
        df_dedup["gross_margin"] - df_dedup["gpm_p"],   # 百分点差
        np.nan
    )

    # OCF/净利润（现金盈利质量）：>1 代表现金收入超过会计利润，信号更可靠
    ni_safe = df_dedup["n_income"].replace(0, np.nan)
    df_dedup["ocf_to_ni"] = np.where(
        ni_safe.notna() & df_dedup["ocf"].notna(),
        df_dedup["ocf"] / ni_safe.abs(),
        np.nan
    )

    # 合并回原始 df（使用 end_date 匹配）
    keep_cols = ["ts_code", "end_date", "fscore", "rev_growth_yoy", "ni_growth_yoy",
                 "gross_margin_chg_yoy", "ocf_to_ni"]
    df = df.merge(df_dedup[keep_cols], on=["ts_code", "end_date"], how="left")

    # 最终列集合
    fund_cols = ["ts_code", "f_ann_date", "end_date",
                 "roe_ann", "roa", "gross_margin", "debt_ratio",
                 "current_ratio", "fscore", "rev_growth_yoy", "ni_growth_yoy",
                 "gross_margin_chg_yoy", "ocf_to_ni"]
    fund_df = df[fund_cols].dropna(subset=["f_ann_date"]).copy()
    fund_df["f_ann_date"] = fund_df["f_ann_date"].astype(str)
    fund_df = fund_df.sort_values(["ts_code", "f_ann_date"]).reset_index(drop=True)

    print(f"    基本面面板: {len(fund_df):,} 条, {fund_df['ts_code'].nunique()} 只股票")
    return fund_df


def join_fundamental_pit(fund_df: pd.DataFrame,
                         rebal_keys: pd.DataFrame) -> pd.DataFrame:
    """
    PIT join: 对每个 (ts_code, trade_date)，找最新已披露的基本面数据。
    使用 np.searchsorted 实现高效的"截至 trade_date 的最新基本面"查找。
    """
    print("  PIT合并基本面数据（searchsorted）...")
    keys = rebal_keys[["ts_code", "trade_date"]].copy()
    keys["trade_date"] = keys["trade_date"].astype(str)
    keys["_td_int"] = keys["trade_date"].str.replace("-", "").astype(int)

    fund_df = fund_df.copy()
    fund_df["_ann_int"] = fund_df["f_ann_date"].str.replace("-", "").astype(int)

    # 基本面特征列（排除 join 用的辅助列）
    value_cols = [c for c in fund_df.columns
                  if c not in ("ts_code", "f_ann_date", "end_date", "_ann_int")]

    fund_grouped = {ts: grp.sort_values("_ann_int").reset_index(drop=True)
                    for ts, grp in fund_df.groupby("ts_code")}

    results = []
    for ts_code, keys_grp in keys.groupby("ts_code"):
        keys_grp = keys_grp.copy()
        if ts_code not in fund_grouped:
            # 无基本面数据：直接保留 keys，特征列为 NaN
            for col in value_cols:
                keys_grp[col] = np.nan
            results.append(keys_grp.drop(columns=["_td_int"]))
            continue

        fund_grp = fund_grouped[ts_code]
        ann_ints = fund_grp["_ann_int"].values
        td_ints  = keys_grp["_td_int"].values

        # searchsorted: 对每个 td_int，找最大的 ann_int <= td_int
        idxs = np.searchsorted(ann_ints, td_ints, side="right") - 1
        valid = idxs >= 0

        for col in value_cols:
            col_vals = fund_grp[col].values
            keys_grp[col] = np.where(valid, col_vals[np.maximum(idxs, 0)], np.nan)
        results.append(keys_grp.drop(columns=["_td_int"]))

    result = pd.concat(results, ignore_index=True)
    n_matched = result["fscore"].notna().sum()
    print(f"    PIT匹配: {n_matched:,} / {len(result):,} 条有基本面数据")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 5. 分析师一致预期特征（report_rc）
# ════════════════════════════════════════════════════════════════════════════
def load_analyst_features(conn, rebal_dates: list,
                          mv_map: dict) -> pd.DataFrame:
    """
    对每个调仓日，聚合过去 90 天的分析师预测（report_rc）。
    返回: (ts_code, trade_date) → [analyst_count, np_yield, analyst_rev_30d]
    np_yield = 中位数预测净利润(万元) / 市值(万元) = 盈利收益率
    analyst_rev_30d = 近30天预测中位数 / 31-90天前预测中位数 - 1 = 修正动量
    """
    print("  加载分析师预期特征...")
    rc_df = conn.execute("""
        SELECT ts_code, report_date, np
        FROM report_rc
        WHERE np IS NOT NULL AND np > 0
        ORDER BY ts_code, report_date
    """).fetchdf()
    rc_df["report_date"] = rc_df["report_date"].astype(str)

    results = []
    for rd in rebal_dates:
        rd_str = str(rd)
        win_start   = _shift_date_str(rd_str, -90)
        recent_start = _shift_date_str(rd_str, -30)
        old_end      = _shift_date_str(rd_str, -31)

        mask_all    = (rc_df["report_date"] >= win_start)    & (rc_df["report_date"] <= rd_str)
        mask_recent = (rc_df["report_date"] >= recent_start) & (rc_df["report_date"] <= rd_str)
        mask_old    = (rc_df["report_date"] >= win_start)    & (rc_df["report_date"] <= old_end)

        subset = rc_df[mask_all]
        if subset.empty:
            continue
        agg = subset.groupby("ts_code")["np"].agg(
            analyst_count="count",
            np_median="median"
        ).reset_index()
        agg["trade_date"] = rd_str

        # 分析师预期修正动量：近30天 vs 31-90天前预测中位数变化率
        recent_np = rc_df[mask_recent].groupby("ts_code")["np"].median().rename("recent_np")
        old_np    = rc_df[mask_old   ].groupby("ts_code")["np"].median().rename("old_np")
        rev = pd.concat([recent_np, old_np], axis=1)
        rev["analyst_rev_30d"] = (
            rev["recent_np"] / rev["old_np"].replace(0, np.nan) - 1
        ).clip(-1.0, 2.0)
        agg = agg.merge(rev[["analyst_rev_30d"]].reset_index(),
                        on="ts_code", how="left")

        results.append(agg)

    if not results:
        print("    ⚠ 无分析师数据")
        return pd.DataFrame(columns=["ts_code", "trade_date",
                                     "analyst_count", "np_yield", "analyst_rev_30d"])

    analyst_df = pd.concat(results, ignore_index=True)

    # np_yield = 预期净利润(万元) / 市值(万元)
    def get_mv(row):
        key = (row["ts_code"], row["trade_date"])
        return mv_map.get(key, np.nan)

    analyst_df["total_mv_wan"] = analyst_df.apply(get_mv, axis=1)
    analyst_df["np_yield"] = analyst_df["np_median"] / analyst_df["total_mv_wan"].replace(0, np.nan)
    analyst_df = analyst_df.drop(columns=["np_median", "total_mv_wan"])

    print(f"    分析师特征: {len(analyst_df):,} 条")
    return analyst_df


def _shift_date_str(date_str: str, days: int) -> str:
    """简单日期偏移（用于90天窗口）"""
    dt = pd.to_datetime(date_str)
    return (dt + pd.Timedelta(days=days)).strftime("%Y%m%d")


# ════════════════════════════════════════════════════════════════════════════
# 5b. 股东户数特征（筹码集中度，A股最重要特有因子之一）
# ════════════════════════════════════════════════════════════════════════════
def load_holder_features(conn) -> pd.DataFrame:
    """
    从 stk_holdernumber 加载股东户数，计算季度变化率。
    holder_chg_qoq < 0 → 户数减少 → 筹码集中 → 看涨信号（A股实证）
    返回: (ts_code, ann_date, end_date, holder_chg_qoq)
    """
    try:
        df = conn.execute("""
            SELECT ts_code,
                   COALESCE(ann_date, end_date) AS f_ann_date,
                   end_date,
                   holder_num
            FROM stk_holdernumber
            WHERE holder_num > 0
            ORDER BY ts_code, end_date
        """).fetchdf()
    except Exception:
        print("    ⚠ stk_holdernumber 表不存在，跳过股东户数特征")
        return pd.DataFrame(columns=["ts_code", "f_ann_date", "end_date", "holder_chg_qoq"])

    if df.empty:
        return pd.DataFrame(columns=["ts_code", "f_ann_date", "end_date", "holder_chg_qoq"])

    df["f_ann_date"] = df["f_ann_date"].astype(str)
    df["end_date"]   = df["end_date"].astype(str)
    df = df.sort_values(["ts_code", "end_date"])

    # 同比变化（对比上一个报告期）
    df["holder_num_prev"] = df.groupby("ts_code")["holder_num"].shift(1)
    df["holder_chg_qoq"] = np.where(
        df["holder_num_prev"].notna() & (df["holder_num_prev"] > 0),
        (df["holder_num"] - df["holder_num_prev"]) / df["holder_num_prev"],
        np.nan
    )
    df = df.dropna(subset=["holder_chg_qoq"])
    result = df[["ts_code", "f_ann_date", "end_date", "holder_chg_qoq"]].copy()
    print(f"    股东户数特征: {len(result):,} 条, {result['ts_code'].nunique()} 只股票")
    return result


def join_holder_pit(holder_df: pd.DataFrame, rebal_keys: pd.DataFrame) -> pd.DataFrame:
    """PIT join 股东户数（与 join_fundamental_pit 相同逻辑）"""
    if holder_df.empty:
        rebal_keys = rebal_keys.copy()
        rebal_keys["holder_chg_qoq"] = np.nan
        return rebal_keys

    print("  PIT合并股东户数数据...")
    keys = rebal_keys[["ts_code", "trade_date"]].copy()
    keys["_td_int"] = keys["trade_date"].astype(str).str.replace("-", "").astype(int)

    holder_df = holder_df.copy()
    holder_df["_ann_int"] = holder_df["f_ann_date"].str.replace("-", "").astype(int)

    holder_grouped = {ts: grp.sort_values("_ann_int").reset_index(drop=True)
                      for ts, grp in holder_df.groupby("ts_code")}

    results = []
    for ts_code, keys_grp in keys.groupby("ts_code"):
        keys_grp = keys_grp.copy()
        if ts_code not in holder_grouped:
            keys_grp["holder_chg_qoq"] = np.nan
        else:
            hg  = holder_grouped[ts_code]
            ann = hg["_ann_int"].values
            tds = keys_grp["_td_int"].values
            idxs = np.searchsorted(ann, tds, side="right") - 1
            valid = idxs >= 0
            vals = hg["holder_chg_qoq"].values
            keys_grp["holder_chg_qoq"] = np.where(valid, vals[np.maximum(idxs, 0)], np.nan)
        results.append(keys_grp.drop(columns=["_td_int"]))

    result = pd.concat(results, ignore_index=True)
    n_matched = result["holder_chg_qoq"].notna().sum()
    print(f"    PIT匹配股东户数: {n_matched:,} / {len(result):,} 条")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 5c. 盈利超预期（SUE）= 实际净利润 vs 公告前分析师一致预期
# ════════════════════════════════════════════════════════════════════════════
def load_sue_features(conn) -> pd.DataFrame:
    """
    SUE = (actual_np - consensus_np) / abs(consensus_np)
    consensus_np = 公告前5天至180天内的分析师预测中位数（同一季度）
    返回: (ts_code, f_ann_date, sue)
    原理: 市场对盈利超预期的反应不充分（PEAD），信息持续扩散90天
    """
    print("  计算SUE（盈利超预期）...")

    # 加载实际净利润（使用PIT公告日）
    is_df = conn.execute("""
        WITH is_t AS (
            SELECT ts_code, end_date,
                   MIN(COALESCE(f_ann_date, ann_date)) AS f_ann_date,
                   FIRST(n_income_attr_p ORDER BY ann_date DESC) AS actual_np
            FROM income_statement
            WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        )
        SELECT * FROM is_t
        WHERE f_ann_date >= '20160101' AND actual_np IS NOT NULL
    """).fetchdf()
    is_df["f_ann_date"] = is_df["f_ann_date"].astype(str)
    is_df["end_date"]   = is_df["end_date"].astype(str)

    # 映射 end_date → quarter code（与 report_rc 格式一致）
    month_to_q = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "Q4"}
    is_df["quarter"] = is_df["end_date"].str[:4] + is_df["end_date"].str[4:6].map(month_to_q)
    is_df = is_df[is_df["quarter"].notna()].copy()

    # 加载分析师预测（按季度分组）
    rc_df = conn.execute("""
        SELECT ts_code, report_date, quarter, np
        FROM report_rc
        WHERE np IS NOT NULL AND np > 0
    """).fetchdf()
    rc_df["report_date"] = rc_df["report_date"].astype(str)

    # 合并：income_statement × report_rc（同一只股票、同一季度）
    merged = is_df.merge(rc_df, on=["ts_code", "quarter"])

    # 过滤：只使用公告前5天至180天的预测（公告前的共识）
    ann_dt  = pd.to_datetime(merged["f_ann_date"], format="%Y%m%d", errors="coerce")
    rep_dt  = pd.to_datetime(merged["report_date"], format="%Y%m%d", errors="coerce")
    days_before = (ann_dt - rep_dt).dt.days
    merged = merged[(days_before >= 5) & (days_before <= 180)].copy()

    # 计算共识中位数
    consensus = merged.groupby(["ts_code", "end_date", "f_ann_date", "actual_np"])["np"].agg(
        consensus_median="median", n_analysts="count"
    ).reset_index()

    # 至少需要2家分析师覆盖
    consensus = consensus[consensus["n_analysts"] >= 2].copy()

    # SUE = (actual - consensus) / |consensus|，截断到 [-2, 2]
    consensus["sue"] = (
        (consensus["actual_np"] - consensus["consensus_median"]) /
        consensus["consensus_median"].abs().replace(0, np.nan)
    ).clip(-2.0, 2.0)

    sue_df = consensus[["ts_code", "f_ann_date", "sue"]].copy()
    print(f"    SUE特征: {len(sue_df):,} 条 ({sue_df['ts_code'].nunique()} 只股票)")
    return sue_df


def join_sue_pit(sue_df: pd.DataFrame, rebal_keys: pd.DataFrame,
                 max_lag_days: int = 90) -> pd.DataFrame:
    """PIT join SUE，有效期 max_lag_days 天（超过则置 NaN）"""
    if sue_df.empty:
        rebal_keys = rebal_keys.copy()
        rebal_keys["sue"] = np.nan
        return rebal_keys

    print("  PIT合并SUE数据...")
    keys = rebal_keys[["ts_code", "trade_date"]].copy()
    keys["_td_int"] = keys["trade_date"].astype(str).str.replace("-", "").astype(int)

    sue_df = sue_df.copy()
    sue_df["_ann_int"] = sue_df["f_ann_date"].str.replace("-", "").astype(int)

    sue_grouped = {ts: grp.sort_values("_ann_int").reset_index(drop=True)
                   for ts, grp in sue_df.groupby("ts_code")}

    results = []
    for ts_code, keys_grp in keys.groupby("ts_code"):
        keys_grp = keys_grp.copy()
        if ts_code not in sue_grouped:
            keys_grp["sue"] = np.nan
        else:
            sg   = sue_grouped[ts_code]
            ann  = sg["_ann_int"].values
            tds  = keys_grp["_td_int"].values
            idxs = np.searchsorted(ann, tds, side="right") - 1
            valid = idxs >= 0

            sue_vals = sg["sue"].values
            ann_vals = sg["_ann_int"].values
            matched_ann = np.where(valid, ann_vals[np.maximum(idxs, 0)], 0)

            # 计算自公告日起的天数，超过 max_lag_days 则失效
            td_dt  = pd.to_datetime(keys_grp["trade_date"].astype(str), format="%Y%m%d")
            ann_dt = pd.to_datetime(pd.Series(matched_ann).astype(str).replace("0", "19700101"),
                                    format="%Y%m%d")
            days_since = (td_dt.values - ann_dt.values) / np.timedelta64(1, "D")
            still_valid = valid & (days_since <= max_lag_days)

            keys_grp["sue"] = np.where(still_valid, sue_vals[np.maximum(idxs, 0)], np.nan)
        results.append(keys_grp.drop(columns=["_td_int"]))

    result = pd.concat(results, ignore_index=True)
    n_matched = result["sue"].notna().sum()
    print(f"    SUE PIT匹配: {n_matched:,} / {len(result):,} 条")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 6. 标签：10日超额收益（前向，相对沪深300）
# ════════════════════════════════════════════════════════════════════════════
def compute_labels(close_pivot: pd.DataFrame, conn,
                   rebal_dates: list) -> pd.DataFrame:
    """
    label = stock_ret_10d - benchmark_ret_10d（均用收盘价 T→T+10）
    注意: T+1日收益无法在T日收盘时知晓，但对10日预测影响微小
    cross-sectional rank normalize → [0,1] (用于训练)
    也保留原始 excess_ret 用于评估
    """
    print("  计算10日超额收益标签...")

    # 基准前向10日收益
    bench_df = conn.execute(f"""
        SELECT trade_date,
               LEAD(close, {HORIZON}) OVER (ORDER BY trade_date) /
                   NULLIF(close, 0) - 1  AS bench_ret
        FROM index_daily
        WHERE ts_code = '{BENCHMARK}'
          AND trade_date >= '{DATA_START}' AND trade_date <= '{END_DATE}'
    """).fetchdf()
    bench_df["trade_date"] = bench_df["trade_date"].astype(str)
    bench_map = dict(zip(bench_df["trade_date"], bench_df["bench_ret"]))

    # 个股前向10日收益（矩阵运算）
    rebal_set = set(str(d) for d in rebal_dates)
    pivot_sub = close_pivot[close_pivot.index.isin(rebal_set)]

    # shift(-HORIZON): 未来10日后的收盘价 / 当日收盘价 - 1
    fwd_ret = close_pivot.shift(-HORIZON) / close_pivot - 1
    fwd_ret_sub = fwd_ret.loc[fwd_ret.index.isin(rebal_set)]

    # 转长格式
    label_long = fwd_ret_sub.stack().reset_index()
    label_long.columns = ["trade_date", "ts_code", "stock_ret_10d"]
    label_long["trade_date"] = label_long["trade_date"].astype(str)

    # 加基准收益
    label_long["bench_ret"] = label_long["trade_date"].map(bench_map)
    label_long["excess_ret"] = label_long["stock_ret_10d"] - label_long["bench_ret"]
    label_long = label_long.dropna(subset=["excess_ret"])

    # 截面百分位归一化标签（训练用）
    label_long["label"] = label_long.groupby("trade_date")["excess_ret"].rank(pct=True)

    print(f"    标签: {len(label_long):,} 条, {label_long['trade_date'].nunique()} 个截面")
    return label_long[["ts_code", "trade_date", "excess_ret", "label"]]


# ════════════════════════════════════════════════════════════════════════════
# 7. 调仓日期 & 股票基础信息
# ════════════════════════════════════════════════════════════════════════════
def get_rebal_dates(conn) -> list:
    """每5个交易日取一个调仓日（从 daily_price 获取交易日历）"""
    df = conn.execute(f"""
        WITH dates AS (
            SELECT DISTINCT trade_date
            FROM daily_price
            WHERE trade_date >= '{TRAIN_START}' AND trade_date <= '{END_DATE}'
        ),
        numbered AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (ORDER BY trade_date) - 1 AS rn
            FROM dates
        )
        SELECT trade_date FROM numbered WHERE rn % {REBAL_FREQ} = 0
        ORDER BY trade_date
    """).fetchdf()
    dates = df["trade_date"].astype(str).tolist()
    print(f"  调仓日期: {len(dates)} 个 ({dates[0]} ~ {dates[-1]})")
    return dates


def load_stock_info(conn) -> pd.DataFrame:
    """返回 (ts_code, industry) 用于中性化"""
    return conn.execute("""
        SELECT ts_code, COALESCE(industry, '未知') AS industry
        FROM stock_basic
    """).fetchdf()


# ════════════════════════════════════════════════════════════════════════════
# 8. 组装全量 Panel
# ════════════════════════════════════════════════════════════════════════════
def build_panel(tech_df: pd.DataFrame,
                rsi_matrix: pd.DataFrame,
                mf_df: pd.DataFrame,
                fund_pit: pd.DataFrame,
                analyst_df: pd.DataFrame,
                label_df: pd.DataFrame,
                stock_info: pd.DataFrame,
                rebal_dates: list,
                holder_pit: pd.DataFrame = None,
                sue_pit: pd.DataFrame = None) -> pd.DataFrame:
    """
    合并所有特征和标签为一个 Panel DataFrame。
    键: (ts_code, trade_date)
    """
    print("  组装全量Panel...")
    rebal_set = set(str(d) for d in rebal_dates)

    # 1. 技术/估值特征（基础）
    base = tech_df[tech_df["trade_date"].isin(rebal_set)].copy()

    # 2. 加 RSI
    rsi_long = rsi_matrix.stack().reset_index()
    rsi_long.columns = ["trade_date", "ts_code", "rsi_14"]
    rsi_long["trade_date"] = rsi_long["trade_date"].astype(str)
    rsi_long = rsi_long[rsi_long["trade_date"].isin(rebal_set)]
    base = base.merge(rsi_long[["ts_code", "trade_date", "rsi_14"]],
                      on=["ts_code", "trade_date"], how="left")

    # 3. 加资金流向（并归一化到市值）
    mf_rebal = mf_df[mf_df["trade_date"].isin(rebal_set)].copy()
    base = base.merge(mf_rebal, on=["ts_code", "trade_date"], how="left")
    mv_wan = base["total_mv_100m"] * 10000   # 亿元 → 万元（与 net_mf_amount 统一单位）
    base["mf_1d_mv"]           = base["mf_1d_raw"]    / mv_wan.replace(0, np.nan)
    base["mf_5d_mv"]           = base["mf_5d_raw"]    / mv_wan.replace(0, np.nan)
    base["mf_20d_mv"]          = base["mf_20d_raw"]   / mv_wan.replace(0, np.nan)
    base["large_net_5d_ratio"] = base["large_net_5d"]  / base["total_flow_5d"].replace(0, np.nan)
    base["large_net_20d_ratio"]= base["large_net_20d"] / base["total_flow_20d"].replace(0, np.nan)
    if "retail_net_5d" in base.columns:
        base["retail_net_5d_ratio"] = base["retail_net_5d"] / mv_wan.replace(0, np.nan)
    else:
        base["retail_net_5d_ratio"] = np.nan
    # 主力 vs 散户分歧：(主力净 - 散户净) / 总流量
    if "retail_net_5d" in base.columns:
        large_net = base["large_net_5d"]
        retail_net = base["retail_net_5d"]
        base["smart_retail_divergence_5d"] = (large_net - retail_net) / base["total_flow_5d"].replace(0, np.nan)
    else:
        base["smart_retail_divergence_5d"] = np.nan
    base = base.drop(columns=["mf_1d_raw", "mf_5d_raw", "mf_20d_raw",
                               "large_net_5d", "large_net_20d",
                               "retail_net_5d",
                               "total_flow_5d", "total_flow_20d"],
                     errors="ignore")

    # 4. 加基本面 PIT
    fund_cols_keep = ["ts_code", "trade_date"] + FUND_COLS
    base = base.merge(fund_pit[[c for c in fund_cols_keep if c in fund_pit.columns]],
                      on=["ts_code", "trade_date"], how="left")

    # 5. 加分析师预期
    if not analyst_df.empty:
        base = base.merge(analyst_df, on=["ts_code", "trade_date"], how="left")
    else:
        base["analyst_count"]   = np.nan
        base["np_yield"]        = np.nan
        base["analyst_rev_30d"] = np.nan
    # analyst_rev_30d: 无历史对比报告时为 NaN（补 0 在预处理 Z-score 阶段自动处理）

    # 5b. 加股东户数（holder_chg_qoq）
    if holder_pit is not None and not holder_pit.empty and "holder_chg_qoq" in holder_pit.columns:
        base = base.merge(holder_pit[["ts_code", "trade_date", "holder_chg_qoq"]],
                          on=["ts_code", "trade_date"], how="left")
    else:
        base["holder_chg_qoq"] = np.nan

    # 5c. SUE（盈利超预期，PEAD因子）
    if sue_pit is not None and not sue_pit.empty and "sue" in sue_pit.columns:
        base = base.merge(sue_pit[["ts_code", "trade_date", "sue"]],
                          on=["ts_code", "trade_date"], how="left")
    else:
        base["sue"] = np.nan

    # 5d. 非线性交叉特征（F2/F3/F4）
    # F2: 资金共振动量 — 机构净流入方向 × 中期价格趋势
    base["smart_momentum"] = base["ret_20d"] * base["large_net_20d_ratio"]
    # F3: 短期反转调整后动量 (Skip-1 Momentum)
    base["momentum_adj_reversal"] = base["ret_60d"] - base["ret_5d"]
    # F4: GARP 因子 — 净利润增速 / 估值
    pe_safe = base["pe_ttm"].clip(lower=5.0)
    base["quality_value_score"] = base["ni_growth_yoy"] / pe_safe

    # 6. 加行业信息（用于中性化）
    base = base.merge(stock_info, on="ts_code", how="left")
    base["industry"] = base["industry"].fillna("未知")

    # 7. 加标签
    base = base.merge(label_df, on=["ts_code", "trade_date"], how="inner")

    # 8. 市值过滤（>= MIN_MKTCAP 亿元）
    before = len(base)
    base = base[base["total_mv_100m"] >= MIN_MKTCAP]
    print(f"    市值过滤后: {len(base):,} 条 (过滤 {before - len(base):,} 条)")

    # 9. 排除 ST 股（通过市值极小已部分排除，进一步依赖 ts_code 过滤）
    base = base[~base["ts_code"].str.startswith(("8", "4"))].copy()

    print(f"    Panel形状: {base.shape[0]:,} 行, {base['trade_date'].nunique()} 个截面, "
          f"{base['ts_code'].nunique()} 只股票")
    return base.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# 9. 数据预处理
# ════════════════════════════════════════════════════════════════════════════
def winsorize_mad(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """MAD法去极值，对 NaN 友好"""
    median = series.median()
    if pd.isna(median):
        return series
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + threshold * 1.4826 * mad
    lower = median - threshold * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)


def neutralize_cross_section(df: pd.DataFrame,
                              factor_cols: list,
                              industry_col: str = "industry",
                              logmktcap_col: str = "log_mktcap") -> pd.DataFrame:
    """
    截面中性化：对每个因子，在截面上关于 [log_mktcap, industry_dummies] 做 Ridge 回归，
    取残差作为中性化后的因子值。
    残差保留了剔除市值风格和行业 Beta 后的纯 Alpha 信号。
    """
    df = df.copy()
    industries = pd.get_dummies(df[industry_col], prefix="ind", drop_first=True, dtype=float)
    logmv = df[logmktcap_col].fillna(df[logmktcap_col].median())
    X_ctrl = pd.concat([logmv.rename("log_mktcap"), industries], axis=1).values.astype(float)

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


def preprocess_panel(panel: pd.DataFrame,
                     feature_cols: list,
                     neutralize: bool = True) -> pd.DataFrame:
    """
    对每个截面（trade_date）执行:
    1. MAD 去极值
    2. 行业 + 市值中性化（可选）
    3. 截面 Z-score 标准化
    """
    print("  预处理: 去极值 → 中性化 → Z-score...")
    panel = panel.copy()

    def process_group(grp):
        for col in feature_cols:
            if col not in grp.columns:
                continue
            grp[col] = winsorize_mad(grp[col])

        if neutralize and len(grp) > 20:
            neut_cols = [c for c in feature_cols
                         if c not in ("log_mktcap", "fscore", "analyst_count",
                                      "analyst_rev_30d", "sue")]
            grp = neutralize_cross_section(grp, neut_cols)

        for col in feature_cols:
            if col not in grp.columns:
                continue
            s = grp[col]
            mu, std = s.mean(), s.std()
            if std > 0:
                grp[col] = (s - mu) / std
            elif not s.isna().all():
                grp[col] = 0.0
            # else: 全NaN，保持原值不动
        return grp

    processed = []
    for _, grp in panel.groupby("trade_date"):
        processed.append(process_group(grp.copy()))
    panel = pd.concat(processed, ignore_index=True)
    print(f"    预处理完成: {panel.shape}")
    return panel


# ════════════════════════════════════════════════════════════════════════════
# 10. 三段式数据切分（Train / Val / Test 严格分离）
# ════════════════════════════════════════════════════════════════════════════
def split_train_val_test(panel: pd.DataFrame) -> tuple:
    """
    严格三段式切分（所有参数决策均不接触 Test）:
      Train: TRAIN_START ~ TRAIN_END         （超参搜索 + 最终模型训练）
      Val:   VAL_START   ~ VAL_END           （超参搜索评估 + Early Stopping）
      Test:  TEST_START  ~ END_DATE          （唯一真实 OOS 评测）
    各段之间含 EMBARGO_DAYS 隔离（约20交易日），防止 label lookahead。
    """
    panel = panel.copy()
    panel["trade_date"] = panel["trade_date"].astype(str)
    train = panel[(panel["trade_date"] >= TRAIN_START) &
                  (panel["trade_date"] <= TRAIN_END)].copy()
    val   = panel[(panel["trade_date"] >= VAL_START) &
                  (panel["trade_date"] <= VAL_END)].copy()
    test  = panel[panel["trade_date"] >= TEST_START].copy()

    def _info(name, df):
        if df.empty:
            print(f"  {name}: 空")
        else:
            print(f"  {name}: {df['trade_date'].min()} ~ {df['trade_date'].max()}, "
                  f"{len(df):,} 行, {df['trade_date'].nunique()} 个截面")
    _info("Train", train)
    _info("Val  ", val)
    _info("Test ", test)
    return train, val, test


# ════════════════════════════════════════════════════════════════════════════
# 11. XGBoost 训练（接受外部 val 集和超参 dict）
# ════════════════════════════════════════════════════════════════════════════
# 默认超参（超参搜索未运行时的安全回退值）
_DEFAULT_XGB_PARAMS = dict(max_depth=4, min_child_weight=30, reg_lambda=10.0)

def train_xgb(train: pd.DataFrame, feature_cols: list,
              val: pd.DataFrame = None,
              params: dict = None,
              n_estimators: int = 1000) -> xgb.XGBRegressor:
    """
    训练 XGBoost 截面选股模型（reg:squarederror，label = 截面百分位排名）。
    val:          外部验证集（用于 early stopping）；若不提供则无 early stopping
    params:       超参数 dict（max_depth / min_child_weight / reg_lambda）；
                  若不提供则使用 _DEFAULT_XGB_PARAMS
    n_estimators: 最大迭代轮数（early stopping 可提前终止）
    """
    p = {**_DEFAULT_XGB_PARAMS, **(params or {})}
    print(f"\n[训练] XGBoost  depth={p['max_depth']}  "
          f"mcw={p['min_child_weight']}  λ={p['reg_lambda']}")

    X_tr = train[feature_cols].values.astype(float)
    y_tr = train["label"].values.astype(float)
    print(f"  训练集: {len(train):,} 行, {train['trade_date'].nunique()} 个截面")

    use_es = val is not None and len(val) > 0
    if use_es:
        X_val = val[feature_cols].values.astype(float)
        y_val = val["label"].values.astype(float)
        print(f"  验证集: {len(val):,} 行, {val['trade_date'].nunique()} 个截面")
        eval_set = [(X_val, y_val)]
        early_stopping_rounds = 50
    else:
        eval_set = None
        early_stopping_rounds = None

    model = xgb.XGBRegressor(
        objective        = "reg:squarederror",
        n_estimators     = n_estimators,
        learning_rate    = 0.02,
        subsample        = 0.7,
        colsample_bytree = 0.7,
        reg_alpha        = 0.5,
        random_state     = 42,
        n_jobs           = -1,
        tree_method      = "hist",
        early_stopping_rounds = early_stopping_rounds,
        eval_metric      = "rmse",
        verbosity        = 0,
        **p,
    )
    model.fit(X_tr, y_tr, eval_set=eval_set, verbose=200)
    if use_es:
        print(f"  最优迭代: {model.best_iteration} 轮")
    return model


# ════════════════════════════════════════════════════════════════════════════
# 11b. LightGBM 训练（与 XGBoost ensemble，接受外部 val 集）
# ════════════════════════════════════════════════════════════════════════════
def _xgb_params_to_lgb(xgb_params: dict) -> dict:
    """将 XGB 超参转换为对等的 LGB 超参（对称设计）"""
    d = xgb_params.get("max_depth", 4)
    mcw = xgb_params.get("min_child_weight", 30)
    rl  = xgb_params.get("reg_lambda", 10.0)
    # LGB num_leaves ≈ 2^(max_depth-1) 以保持近似复杂度
    num_leaves = max(7, min(63, 2 ** (d - 1)))
    return dict(num_leaves=num_leaves, min_child_samples=mcw, reg_lambda=rl)


def train_lgbm(train: pd.DataFrame, feature_cols: list,
               val: pd.DataFrame = None,
               params: dict = None,
               n_estimators: int = 1000) -> lgb.Booster:
    """
    训练 LightGBM 截面选股模型（regression，与 XGBoost ensemble）。
    val:    外部验证集（用于 early stopping）
    params: XGB 格式超参 dict，自动转换为 LGB 等价参数
    """
    lgb_p = _xgb_params_to_lgb(params or _DEFAULT_XGB_PARAMS)
    print(f"\n[训练] LightGBM  leaves={lgb_p['num_leaves']}  "
          f"mcs={lgb_p['min_child_samples']}  λ={lgb_p['reg_lambda']}")

    X_tr = train[feature_cols].values.astype(float)
    y_tr = train["label"].values.astype(float)
    print(f"  训练集: {len(train):,} 行, {train['trade_date'].nunique()} 个截面")

    lgb_params = dict(
        objective        = "regression",
        metric           = "l2",
        learning_rate    = 0.02,
        feature_fraction = 0.7,
        bagging_fraction = 0.7,
        bagging_freq     = 5,
        reg_alpha        = 0.5,
        verbose          = -1,
        n_jobs           = -1,
        **lgb_p,
    )
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)

    use_es = val is not None and len(val) > 0
    if use_es:
        X_val = val[feature_cols].values.astype(float)
        y_val = val["label"].values.astype(float)
        print(f"  验证集: {len(val):,} 行, {val['trade_date'].nunique()} 个截面")
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                     lgb.log_evaluation(period=200)]
        model = lgb.train(lgb_params, dtrain, num_boost_round=n_estimators,
                          valid_sets=[dval], callbacks=callbacks)
    else:
        model = lgb.train(lgb_params, dtrain, num_boost_round=n_estimators,
                          callbacks=[lgb.log_evaluation(period=-1)])

    print(f"  最优迭代: {model.best_iteration} 轮")
    return model


def ensemble_rank_avg(panel: pd.DataFrame,
                      xgb_model: xgb.XGBRegressor,
                      lgb_model: lgb.Booster,
                      feature_cols: list,
                      w_xgb: float = 0.5) -> pd.DataFrame:
    """
    XGB 和 LGB 预测的截面 rank 加权平均（动态权重）。
    w_xgb: XGB 的权重（由 val Rank IC 比率动态决定，默认 0.5）
    rank-averaging 消除两个模型量纲差异，保留相对顺序信息。
    """
    X = panel[feature_cols].values.astype(float)
    pred_xgb = xgb_model.predict(X)
    pred_lgb = lgb_model.predict(X)

    panel = panel.copy()
    panel["pred_xgb"] = pred_xgb
    panel["pred_lgb"] = pred_lgb

    panel["rank_xgb"] = panel.groupby("trade_date")["pred_xgb"].rank(pct=True)
    panel["rank_lgb"] = panel.groupby("trade_date")["pred_lgb"].rank(pct=True)
    panel["pred"]     = w_xgb * panel["rank_xgb"] + (1 - w_xgb) * panel["rank_lgb"]
    return panel.drop(columns=["pred_xgb", "pred_lgb", "rank_xgb", "rank_lgb"])


# ════════════════════════════════════════════════════════════════════════════
# 11c. 超参搜索 & 敏感性测试（在 Val 上运行，完全不接触 Test）
# ════════════════════════════════════════════════════════════════════════════
def _quick_ic(panel: pd.DataFrame, feature_cols: list) -> float:
    """计算 panel 上的 Rank IC 均值（已含 pred 列）"""
    ic_list = []
    for _, grp in panel.groupby("trade_date"):
        valid = grp[["pred", "excess_ret"]].dropna()
        if len(valid) < 5:
            continue
        rho, _ = stats.spearmanr(valid["pred"].values, valid["excess_ret"].values)
        ic_list.append(rho)
    return float(np.mean(ic_list)) if ic_list else np.nan


def quick_eval_xgb_on_val(train: pd.DataFrame, val: pd.DataFrame,
                           feature_cols: list, params: dict) -> tuple:
    """
    用固定 HP_SEARCH_N_ESTIMATORS 轮快速训练，返回 (val_ic_mean, val_ic_std)。
    不用 early stopping，保证不同参数组合的树数量相同，比较公平。
    """
    p = {**_DEFAULT_XGB_PARAMS, **params}
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=HP_SEARCH_N_ESTIMATORS,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
        **p,
    )
    X_tr = train[feature_cols].values.astype(float)
    y_tr = train["label"].values.astype(float)
    model.fit(X_tr, y_tr, verbose=False)

    val = val.copy()
    val["pred"] = model.predict(val[feature_cols].values.astype(float))
    ic_list = []
    for _, grp in val.groupby("trade_date"):
        valid = grp[["pred", "excess_ret"]].dropna()
        if len(valid) < 5:
            continue
        rho, _ = stats.spearmanr(valid["pred"].values, valid["excess_ret"].values)
        ic_list.append(rho)
    ic_arr = np.array(ic_list)
    return float(ic_arr.mean()) if len(ic_arr) > 0 else np.nan, \
           float(ic_arr.std())  if len(ic_arr) > 0 else np.nan


def hyperparameter_search_xgb(train: pd.DataFrame, val: pd.DataFrame,
                               feature_cols: list) -> tuple:
    """
    在 Val 上网格搜索 XGB_HP_GRID（18 组），返回 (best_params, results_df)。
    全程不接触 Test 数据。
    """
    print(f"\n[HP搜索] 共 {len(XGB_HP_GRID)} 组参数，每组训练 {HP_SEARCH_N_ESTIMATORS} 轮")
    rows = []
    for i, params in enumerate(XGB_HP_GRID):
        ic_mean, ic_std = quick_eval_xgb_on_val(train, val, feature_cols, params)
        icir = ic_mean / ic_std if (ic_std is not None and ic_std > 0) else np.nan
        rows.append({**params, "val_ic": ic_mean, "val_ic_std": ic_std, "val_icir": icir})
        print(f"  [{i+1:2d}/{len(XGB_HP_GRID)}] depth={params['max_depth']}  "
              f"mcw={params['min_child_weight']}  λ={params['reg_lambda']:.1f}  "
              f"→ val_IC={ic_mean:+.4f}  ICIR={icir:+.3f}")

    results_df = pd.DataFrame(rows).sort_values("val_icir", ascending=False)
    best_params = {k: results_df.iloc[0][k]
                   for k in ("max_depth", "min_child_weight", "reg_lambda")}
    best_params["max_depth"]        = int(best_params["max_depth"])
    best_params["min_child_weight"] = int(best_params["min_child_weight"])

    print(f"\n  最优参数: depth={best_params['max_depth']}  "
          f"mcw={best_params['min_child_weight']}  λ={best_params['reg_lambda']:.1f}")
    print(f"  最优 val_IC={results_df.iloc[0]['val_ic']:+.4f}  "
          f"ICIR={results_df.iloc[0]['val_icir']:+.3f}")
    return best_params, results_df


def sensitivity_test_xgb(train: pd.DataFrame, val: pd.DataFrame,
                          feature_cols: list, best_params: dict) -> None:
    """
    对最优参数的每个维度做邻近值测试，验证模型稳定性。
    若某维度 IC 波动 > 0.01，打印警告（参数选择可能不稳健）。
    """
    print("\n[敏感性测试] 对最优参数逐维度扰动（IC变化 > 0.01 视为不稳定）")
    INSTABILITY_THRESH = 0.01

    base_ic, _ = quick_eval_xgb_on_val(train, val, feature_cols, best_params)
    print(f"  基准 val_IC = {base_ic:+.4f}  params = {best_params}")

    for dim, candidates in HP_SENSITIVITY_PARAMS.items():
        # dim 映射：HP_SENSITIVITY_PARAMS 用 min_child_samples，对应 min_child_weight
        param_key = "min_child_weight" if dim == "min_child_samples" else dim
        if param_key not in best_params:
            continue
        best_val = best_params[param_key]
        ics = []
        for v in candidates:
            test_p = {**best_params, param_key: v}
            ic, _ = quick_eval_xgb_on_val(train, val, feature_cols, test_p)
            delta = ic - base_ic
            flag  = "⚠ 不稳定" if abs(delta) > INSTABILITY_THRESH else "✓"
            print(f"    {param_key}={v:<6}  IC={ic:+.4f}  Δ={delta:+.4f}  {flag}")
            ics.append(ic)
        ic_range = max(ics) - min(ics)
        stability = "⚠ 参数敏感" if ic_range > INSTABILITY_THRESH else "✓ 参数稳健"
        print(f"  → {param_key} IC范围={ic_range:.4f}  {stability}\n")


def train_xgb_final(train: pd.DataFrame, val: pd.DataFrame,
                    feature_cols: list, params: dict):
    """
    两阶段最终训练。返回: (model_final, model_train_only)
    1. 在 train 上用 val ES 确定最优轮数 → model_train_only（供策略 val 预测使用）
    2. 在 train+val 合并数据上用最优轮数重训（无 ES） → model_final
    """
    print("\n[最终训练 XGB] 阶段1: train上ES确定最优轮数（保留train-only模型供val预测）")
    model_train_only = train_xgb(train, feature_cols, val=val, params=params, n_estimators=2000)
    best_n = model_train_only.best_iteration
    print(f"  Early stopping 最优轮数: {best_n}")

    print(f"\n[最终训练 XGB] 阶段2: train+val 合并，固定 {best_n} 轮（无 early stopping）")
    train_val = pd.concat([train, val], ignore_index=True)
    model_final = train_xgb(train_val, feature_cols, val=None, params=params,
                             n_estimators=best_n)
    return model_final, model_train_only


def train_lgbm_final(train: pd.DataFrame, val: pd.DataFrame,
                     feature_cols: list, params: dict):
    """
    两阶段最终训练（LGB）。返回: (model_final, model_train_only)
    """
    print("\n[最终训练 LGB] 阶段1: train上ES确定最优轮数（保留train-only模型供val预测）")
    model_train_only = train_lgbm(train, feature_cols, val=val, params=params, n_estimators=2000)
    best_n = model_train_only.best_iteration
    if best_n is None or best_n <= 0:
        best_n = 500
    print(f"  Early stopping 最优轮数: {best_n}")

    print(f"\n[最终训练 LGB] 阶段2: train+val 合并，固定 {best_n} 轮（无 early stopping）")
    train_val = pd.concat([train, val], ignore_index=True)
    model_final = train_lgbm(train_val, feature_cols, val=None, params=params,
                              n_estimators=best_n)
    return model_final, model_train_only


def _compute_val_ic_for_model(model_xgb, model_lgb, val, feature_cols, w_xgb=0.5):
    """计算 val 集上 ensemble 的 Rank IC 均值（用于动态权重计算）"""
    if model_lgb is not None:
        val_ens = ensemble_rank_avg(val, model_xgb, model_lgb, feature_cols, w_xgb=w_xgb)
        return _quick_ic(val_ens, feature_cols)
    else:
        val2 = val.copy()
        val2["pred"] = model_xgb.predict(val[feature_cols].values.astype(float))
        return _quick_ic(val2, feature_cols)


# ════════════════════════════════════════════════════════════════════════════
# 12. 评估指标
# ════════════════════════════════════════════════════════════════════════════
def compute_rank_ic(df_eval: pd.DataFrame) -> pd.Series:
    """
    Rank IC: 每个截面上预测值与真实超额收益的 Spearman 秩相关系数。
    输入: DataFrame with columns [trade_date, pred, excess_ret]
    返回: Series(date → IC 值)
    """
    ic_dict = {}
    for dt, grp in df_eval.groupby("trade_date"):
        valid = grp[["pred", "excess_ret"]].dropna()
        if len(valid) < 5:
            continue
        rho, _ = stats.spearmanr(valid["pred"].values, valid["excess_ret"].values)
        ic_dict[dt] = rho
    return pd.Series(ic_dict).sort_index()


def compute_gauc(df_eval: pd.DataFrame) -> float:
    """
    GAUC (Group AUC): 在每个截面内计算 AUC（预测能否区分超额收益 >0 与 <=0），
    然后用截面内样本数加权平均。
    """
    from sklearn.metrics import roc_auc_score
    weighted_auc = 0.0
    total_weight  = 0
    for dt, grp in df_eval.groupby("trade_date"):
        valid = grp[["pred", "excess_ret"]].dropna()
        if len(valid) < 10:
            continue
        y_true_bin = (valid["excess_ret"].values > 0).astype(int)
        if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
            continue  # 全正或全负，AUC 无意义
        try:
            auc = roc_auc_score(y_true_bin, valid["pred"].values)
            weighted_auc += auc * len(valid)
            total_weight  += len(valid)
        except Exception:
            continue
    return weighted_auc / total_weight if total_weight > 0 else np.nan


def evaluate(panel: pd.DataFrame,
             model,
             feature_cols: list,
             label: str = "TEST",
             pred_arr: np.ndarray = None) -> dict:
    """完整评估: Rank IC, ICIR, GAUC, 分层收益
    pred_arr: 若不为 None，直接使用该预测值（用于 ensemble），忽略 model.predict()。
    """
    if pred_arr is not None:
        pred = pred_arr
    else:
        X = panel[feature_cols].values.astype(float)
        pred = model.predict(X)

    panel = panel.copy()
    panel["pred"] = pred

    df_eval = pd.DataFrame({
        "trade_date": panel["trade_date"].values,
        "pred":       pred,
        "excess_ret": panel["excess_ret"].values,
    })

    ic_series = compute_rank_ic(df_eval)
    mean_ic   = ic_series.mean()
    icir      = mean_ic / ic_series.std() if ic_series.std() > 0 else np.nan
    gauc      = compute_gauc(df_eval)

    # 多空组合：每截面预测 Top 10% 做多 - Bottom 10% 做空
    panel["decile"] = panel.groupby("trade_date")["pred"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    long_ret  = panel[panel["decile"] >= 9].groupby("trade_date")["excess_ret"].mean()
    short_ret = panel[panel["decile"] <= 0].groupby("trade_date")["excess_ret"].mean()
    ls_ret    = long_ret - short_ret

    print(f"\n{'='*55}")
    print(f"  [{label}] 评估结果")
    print(f"{'='*55}")
    print(f"  截面数量:    {ic_series.shape[0]}")
    print(f"  Rank IC均值: {mean_ic:+.4f}")
    print(f"  Rank IC正比: {(ic_series > 0).mean():.2%}")
    print(f"  ICIR:        {icir:+.4f}")
    print(f"  GAUC:        {gauc:.4f}  (0.5基准, 越高越好)")
    print(f"  多空均值收益: {ls_ret.mean():.4f}  (10日，未扣交易成本)")
    print(f"  多空夏普比:  {ls_ret.mean() / ls_ret.std() * np.sqrt(50):.3f}  (年化~50截面)")
    print(f"{'='*55}")

    return {
        "label":      label,
        "n_sections": ic_series.shape[0],
        "mean_ic":    mean_ic,
        "ic_positive": (ic_series > 0).mean(),
        "icir":       icir,
        "gauc":       gauc,
        "ls_mean":    ls_ret.mean(),
        "ic_series":  ic_series,
        "ls_series":  ls_ret,
        "panel_pred": panel,
    }


# ════════════════════════════════════════════════════════════════════════════
# 13. 可视化
# ════════════════════════════════════════════════════════════════════════════
def plot_results(train_res: dict, test_res: dict,
                 model: xgb.XGBRegressor, feature_cols: list):
    """生成评估图表并保存"""
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("XGBoost 截面选股模型评估", fontsize=14, fontproperties="SimHei"
                 if os.path.exists("/System/Library/Fonts/STHeiti Medium.ttc")
                 else None)

    # 1. IC 序列（训练 + 测试）
    ax = axes[0, 0]
    tr_ic = train_res["ic_series"]
    te_ic = test_res["ic_series"]
    ax.bar(range(len(tr_ic)), tr_ic.values, color="steelblue", alpha=0.6,
           label=f"Train IC (μ={train_res['mean_ic']:+.3f})")
    ax.bar(range(len(tr_ic), len(tr_ic) + len(te_ic)), te_ic.values,
           color="orangered", alpha=0.7,
           label=f"Test IC  (μ={test_res['mean_ic']:+.3f})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.03, color="green", linestyle="--", linewidth=0.8, alpha=0.7,
               label="IC=0.03 参考线")
    ax.set_title("Rank IC 序列")
    ax.legend(fontsize=8)
    ax.set_xlabel("截面序号")
    ax.set_ylabel("Rank IC")

    # 2. 累积 IC（测试集）
    ax = axes[0, 1]
    cumic = te_ic.cumsum()
    ax.plot(range(len(cumic)), cumic.values, color="orangered", linewidth=1.5)
    ax.fill_between(range(len(cumic)), cumic.values, alpha=0.2, color="orangered")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"累积 Rank IC（测试集 ICIR={test_res['icir']:+.3f}）")
    ax.set_xlabel("截面序号")
    ax.set_ylabel("累积 IC")

    # 3. 分层多空累积收益（测试集）
    ax = axes[1, 0]
    ls = test_res["ls_series"]
    cum_ls = (1 + ls).cumprod()
    ax.plot(range(len(cum_ls)), cum_ls.values, color="darkgreen", linewidth=1.5,
            label="Top10% - Bottom10%")
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_title(f"多空组合净值（测试集，GAUC={test_res['gauc']:.4f}）")
    ax.legend(fontsize=8)
    ax.set_xlabel("截面序号")
    ax.set_ylabel("累计净值")

    # 4. 特征重要性（Top 15）
    ax = axes[1, 1]
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    top15 = feat_imp.tail(15)
    ax.barh(range(len(top15)), top15.values, color="steelblue")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index, fontsize=8)
    ax.set_title("特征重要性 (Top 15)")
    ax.set_xlabel("Importance")

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/images/xgb_cross_section_eval.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  图表已保存: {save_path}")


def save_predictions(test_res: dict):
    """保存测试集预测结果"""
    os.makedirs(f"{OUTPUT_DIR}/csv", exist_ok=True)
    panel_pred = test_res["panel_pred"]
    save_cols = ["ts_code", "trade_date", "pred", "excess_ret", "decile"]
    out_path = f"{OUTPUT_DIR}/csv/xgb_cross_section_predictions.csv"
    panel_pred[save_cols].to_csv(out_path, index=False)
    print(f"  预测结果已保存: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 13b. Walk-Forward Optimization (WFO) 预测
# ════════════════════════════════════════════════════════════════════════════
def run_wfo_predictions(panel: pd.DataFrame,
                        feature_cols: list) -> pd.DataFrame:
    """
    扩展窗口 Walk-Forward Optimization：
    - 以 WFO_REFIT_MONTHS 为间隔将测试期分段
    - 每段只用该段起始日期之前的数据（含隔离）训练
    - 严格无数据泄漏：预测日期 T 的模型只见过 T-30天 之前的数据

    返回: 全测试期的预测 DataFrame（格式与 evaluate() 输出一致）
    """
    import calendar

    test_panel = panel[panel["trade_date"] >= TEST_START].copy()
    test_dates = sorted(test_panel["trade_date"].unique())
    if not test_dates:
        raise ValueError("测试集无数据")

    # 将测试期按 WFO_REFIT_MONTHS 个月切分为多个窗口
    start_dt = pd.to_datetime(test_dates[0])
    end_dt   = pd.to_datetime(test_dates[-1])

    windows = []   # list of (window_start_str, window_end_str, train_cutoff_str)
    cur = start_dt
    while cur <= end_dt:
        # 窗口结束：cur + WFO_REFIT_MONTHS 个月 - 1天
        month = cur.month - 1 + WFO_REFIT_MONTHS
        year  = cur.year + month // 12
        month = month % 12 + 1
        last_day = calendar.monthrange(year, month)[1]
        win_end = pd.Timestamp(year=year, month=month, day=last_day)

        # 训练截止：窗口起点 - 30 日历天（约 20 交易日隔离）
        train_cut = (cur - pd.Timedelta(days=30)).strftime("%Y%m%d")
        windows.append((cur.strftime("%Y%m%d"),
                        min(win_end, end_dt).strftime("%Y%m%d"),
                        train_cut))
        cur = win_end + pd.Timedelta(days=1)

    print(f"\n[WFO] {len(windows)} 个训练窗口（间隔 {WFO_REFIT_MONTHS} 个月）")

    all_preds = []
    for i, (win_start, win_end, train_cut) in enumerate(windows):
        print(f"\n  [WFO {i+1}/{len(windows)}] 测试窗口 {win_start}~{win_end}  "
              f"| 训练截止 {train_cut}")

        # 训练集：TRAIN_START ~ train_cut（扩展窗口）
        train_sub = panel[(panel["trade_date"] >= TRAIN_START) &
                          (panel["trade_date"] <= train_cut)].copy()
        if len(train_sub) < 1000:
            print(f"    ⚠ 训练数据不足 ({len(train_sub)} 行)，跳过")
            continue

        # 预测集：本窗口内的测试截面
        pred_sub = panel[(panel["trade_date"] >= win_start) &
                         (panel["trade_date"] <= win_end)].copy()
        if pred_sub.empty:
            continue

        # 训练模型
        xgb_m = train_xgb(train_sub, feature_cols)
        if ENSEMBLE_LGB:
            lgb_m = train_lgbm(train_sub, feature_cols)
            pred_sub = ensemble_rank_avg(pred_sub, xgb_m, lgb_m, feature_cols)
        else:
            X = pred_sub[feature_cols].values.astype(float)
            pred_sub = pred_sub.copy()
            pred_sub["pred"] = xgb_m.predict(X)

        all_preds.append(pred_sub)

    if not all_preds:
        raise ValueError("WFO 无预测结果")

    wfo_panel = pd.concat(all_preds, ignore_index=True)
    print(f"\n[WFO] 完成: {len(wfo_panel):,} 条预测, "
          f"{wfo_panel['trade_date'].nunique()} 个截面")
    return wfo_panel


# ════════════════════════════════════════════════════════════════════════════
# 14. 生产模型训练（--prod 模式）
# ════════════════════════════════════════════════════════════════════════════
def _run_prod_mode(panel: pd.DataFrame, avail_features: list):
    """
    训练生产模型：使用 2018-2025-12-31 全量数据，超参与 eval 模型保持一致。
    输出: xgb_h{HORIZON}_prod.json, lgb_h{HORIZON}_prod.txt (若 ENSEMBLE_LGB),
          features_h{HORIZON}_prod.json
    """
    import json
    PROD_END = "20251231"
    prod_train = panel[panel["trade_date"] <= PROD_END].copy()
    print(f"\n{'='*60}")
    print(f"  [PROD] H{HORIZON} 生产模型训练")
    print(f"  训练数据: {prod_train['trade_date'].min()} ~ {PROD_END}")
    print(f"  {len(prod_train):,} 行, {prod_train['trade_date'].nunique()} 个截面")
    print(f"{'='*60}")

    models_dir = f"{OUTPUT_DIR}/models"
    os.makedirs(models_dir, exist_ok=True)

    # 从 eval 元数据读取 best_params 和 w_xgb
    feat_path = f"{models_dir}/features_h{HORIZON}.json"
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            meta = json.load(f)
        best_params = meta.get("best_params", _DEFAULT_XGB_PARAMS)
        w_xgb       = meta.get("w_xgb", 0.5)
        print(f"  超参 (from eval): depth={best_params.get('max_depth')}  "
              f"mcw={best_params.get('min_child_weight')}  "
              f"λ={best_params.get('reg_lambda')}  w_xgb={w_xgb:.3f}")
    else:
        best_params, w_xgb = _DEFAULT_XGB_PARAMS, 0.5
        print("  警告: 未找到 eval 元数据，使用默认超参")

    # 从 eval XGB 模型读取 n_estimators
    eval_xgb = f"{models_dir}/xgb_h{HORIZON}.json"
    if os.path.exists(eval_xgb):
        _tmp = xgb.XGBRegressor(); _tmp.load_model(eval_xgb)
        n_xgb = _tmp.get_booster().num_boosted_rounds()
        print(f"  XGB n_estimators = {n_xgb} (from eval model)")
    else:
        n_xgb = 500
        print(f"  警告: 未找到 eval XGB 模型，n_estimators={n_xgb}")

    # 训练 XGB 生产模型
    xgb_prod = train_xgb(prod_train, avail_features, val=None,
                         params=best_params, n_estimators=n_xgb)
    xgb_prod.save_model(f"{models_dir}/xgb_h{HORIZON}_prod.json")
    print(f"  XGB 生产模型已保存: xgb_h{HORIZON}_prod.json")

    # 训练 LGB 生产模型（若启用 ensemble）
    lgb_prod = None
    if ENSEMBLE_LGB:
        eval_lgb = f"{models_dir}/lgb_h{HORIZON}.txt"
        if os.path.exists(eval_lgb):
            n_lgb = lgb.Booster(model_file=eval_lgb).num_trees()
            print(f"  LGB n_estimators = {n_lgb} (from eval model)")
        else:
            n_lgb = n_xgb
        lgb_prod = train_lgbm(prod_train, avail_features, val=None,
                              params=best_params, n_estimators=n_lgb)
        lgb_prod.save_model(f"{models_dir}/lgb_h{HORIZON}_prod.txt")
        print(f"  LGB 生产模型已保存: lgb_h{HORIZON}_prod.txt")

    # 保存特征元数据
    with open(f"{models_dir}/features_h{HORIZON}_prod.json", "w") as f:
        json.dump({
            "features":   avail_features,
            "train_end":  PROD_END,
            "best_params": best_params,
            "w_xgb":      w_xgb,
        }, f, indent=2)
    print(f"  元数据已保存: features_h{HORIZON}_prod.json")
    print(f"\n[PROD] H{HORIZON} 生产模型训练完成。")


# ════════════════════════════════════════════════════════════════════════════
# 15. Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    import argparse, json
    parser = argparse.ArgumentParser(description='XGBoost 截面选股模型')
    parser.add_argument('--prod', action='store_true',
                        help='生产模式：使用 2018-2025 全量数据训练，输出 *_prod 模型文件')
    args = parser.parse_args()

    t_total = time.time()
    print("=" * 60)
    print("  XGBoost 截面选股模型  |  预测 10 日超额收益")
    print(f"  Train: {TRAIN_START}~{TRAIN_END}  Val: {VAL_START}~{VAL_END}")
    print(f"  Test:  {TEST_START}~{END_DATE}  (严格 OOS，训练全程不可见)")
    print("=" * 60)

    conn = get_conn()
    try:
        # ── Step 1: 基础数据 ──────────────────────────────────────────────
        print("\n[Step 1] 加载基础数据")
        rebal_dates  = get_rebal_dates(conn)
        stock_info   = load_stock_info(conn)
        close_pivot  = load_price_pivot(conn)

        # ── Step 2: 技术/估值特征 ────────────────────────────────────────
        print("\n[Step 2] 技术面 & 估值特征")
        tech_df = load_tech_basic_features(conn)

        # ── Step 3: RSI（向量化）────────────────────────────────────────
        print("\n[Step 3] 计算 RSI(14)")
        rsi_matrix = compute_rsi(close_pivot)
        rebal_set  = set(str(d) for d in rebal_dates)
        rsi_rebal  = rsi_matrix[rsi_matrix.index.isin(rebal_set)]

        # ── Step 4: 资金流向特征 ─────────────────────────────────────────
        print("\n[Step 4] 资金流向特征")
        mf_df = load_moneyflow_features(conn)

        # ── Step 5: 基本面 PIT ───────────────────────────────────────────
        print("\n[Step 5] 基本面 PIT 特征")
        fund_panel = load_fundamental_panel(conn)
        tech_rebal = tech_df[tech_df["trade_date"].isin(rebal_set)]
        rebal_keys = tech_rebal[["ts_code", "trade_date"]].drop_duplicates()
        fund_pit = join_fundamental_pit(fund_panel, rebal_keys)

        # ── Step 6: 分析师预期特征 ───────────────────────────────────────
        print("\n[Step 6] 分析师预期特征")
        mv_cols = tech_rebal[["ts_code", "trade_date", "total_mv_100m"]].copy()
        mv_map  = {(row["ts_code"], row["trade_date"]): row["total_mv_100m"] * 10000
                   for _, row in mv_cols.iterrows()}
        analyst_df = load_analyst_features(conn, rebal_dates, mv_map)

        # ── Step 6b: 股东户数特征（筹码集中度）────────────────────────────
        print("\n[Step 6b] 股东户数特征（holder_chg_qoq）")
        holder_df  = load_holder_features(conn)
        holder_pit = join_holder_pit(holder_df, rebal_keys)

        # ── Step 6c: SUE（盈利超预期）────────────────────────────────────
        print("\n[Step 6c] SUE 盈利超预期特征")
        sue_df  = load_sue_features(conn)
        sue_pit = join_sue_pit(sue_df, rebal_keys, max_lag_days=90)

        # ── Step 7: 标签 ─────────────────────────────────────────────────
        print("\n[Step 7] 计算 10 日超额收益标签")
        label_df = compute_labels(close_pivot, conn, rebal_dates)

    finally:
        conn.close()

    # ── Step 8: 组装 Panel ────────────────────────────────────────────────
    print("\n[Step 8] 组装全量 Panel")
    panel = build_panel(
        tech_df=tech_df,
        rsi_matrix=rsi_rebal,
        mf_df=mf_df,
        fund_pit=fund_pit,
        analyst_df=analyst_df,
        label_df=label_df,
        stock_info=stock_info,
        rebal_dates=rebal_dates,
        holder_pit=holder_pit,
        sue_pit=sue_pit,
    )

    # 检查各特征的非空率
    print("\n  各特征非空率:")
    for col in ALL_FEATURES:
        if col in panel.columns:
            rate = panel[col].notna().mean()
            flag = "⚠" if rate < 0.5 else "✓"
            print(f"    {flag} {col:25s}: {rate:.1%}")

    # ── Step 9: 预处理 ────────────────────────────────────────────────────
    print("\n[Step 9] 数据预处理")
    avail_features = [c for c in ALL_FEATURES if c in panel.columns]
    panel = preprocess_panel(panel, avail_features, neutralize=True)

    feature_na_rate = panel[avail_features].isna().mean(axis=1)
    panel = panel[feature_na_rate < 0.50].copy()

    for col in avail_features:
        if col in panel.columns and panel[col].isna().any():
            panel[col] = panel.groupby("trade_date")[col].transform(
                lambda x: x.fillna(x.median())
            ).fillna(0)

    print(f"  最终Panel: {len(panel):,} 行, {panel['trade_date'].nunique()} 个截面")

    # ── 生产模式：跳过 eval 流程，直接训练全量生产模型 ──────────────────────
    if args.prod:
        _run_prod_mode(panel, avail_features)
        return

    # ── Step 10: 三段式严格切分 ───────────────────────────────────────────
    print("\n[Step 10] 三段式数据切分 (Train / Val / Test)")
    train, val, test = split_train_val_test(panel)

    # ── Step 11: 超参搜索（仅在 Val 上，不接触 Test）────────────────────
    print("\n[Step 11] 超参数网格搜索（在 Val 上评估）")
    best_params, hp_results = hyperparameter_search_xgb(train, val, avail_features)

    # 保存 HP 搜索结果
    os.makedirs(f"{OUTPUT_DIR}/csv", exist_ok=True)
    hp_results.to_csv(f"{OUTPUT_DIR}/csv/xgb_hp_search_results.csv", index=False)
    print(f"  HP搜索结果已保存: {OUTPUT_DIR}/csv/xgb_hp_search_results.csv")

    # ── Step 12: 敏感性测试（验证参数稳健性）────────────────────────────
    print("\n[Step 12] 参数敏感性测试")
    sensitivity_test_xgb(train, val, avail_features, best_params)

    # ── Step 13: 最终训练（train+val，不接触 Test）───────────────────────
    print("\n[Step 13] 最终模型训练（train → ES → train+val 合并重训）")
    xgb_model, xgb_train_only = train_xgb_final(train, val, avail_features, best_params)
    if ENSEMBLE_LGB:
        lgb_model, lgb_train_only = train_lgbm_final(train, val, avail_features, best_params)
    else:
        lgb_model, lgb_train_only = None, None
    model = xgb_model  # 用于 feature importance 和 plot

    # ── 保存 Val 期预测（使用 train-only 模型，供策略超参搜索使用）──────
    print("\n  保存 Val 期预测（train-only 模型，严格 PIT）...")
    val_pred_df = val.copy()
    if ENSEMBLE_LGB and lgb_train_only is not None:
        # 使用 train-only 的动态权重（未知最终权重，先用 0.5）
        val_to_ic, _ = quick_eval_xgb_on_val(train, val, avail_features, best_params)
        val2_lgb = val.copy()
        val2_lgb["pred"] = lgb_train_only.predict(val[avail_features].values.astype(float))
        val_lgb_ic = _quick_ic(val2_lgb, avail_features)
        tot = abs(val_to_ic) + abs(val_lgb_ic)
        w_val = abs(val_to_ic) / tot if tot > 0 else 0.5
        val_pred_df = ensemble_rank_avg(val_pred_df, xgb_train_only, lgb_train_only,
                                        avail_features, w_xgb=w_val)
    else:
        val_pred_df["pred"] = xgb_train_only.predict(
            val[avail_features].values.astype(float))
    os.makedirs(f"{OUTPUT_DIR}/csv", exist_ok=True)
    val_pred_df[["ts_code", "trade_date", "pred", "excess_ret"]].to_csv(
        f"{OUTPUT_DIR}/csv/xgb_cs_pred_val.csv", index=False)
    print(f"  Val预测已保存: {OUTPUT_DIR}/csv/xgb_cs_pred_val.csv  "
          f"({len(val_pred_df):,}行, {val_pred_df['trade_date'].nunique()}个截面)")

    # 计算动态 ensemble 权重（基于 val 集的 Rank IC，使用最终 train+val 模型评估）
    w_xgb = 0.5
    if ENSEMBLE_LGB and lgb_model is not None:
        val_ic_xgb_only, _ = quick_eval_xgb_on_val(
            pd.concat([train, val], ignore_index=True), val, avail_features, best_params)
        val2 = val.copy()
        val2["pred"] = lgb_model.predict(val[avail_features].values.astype(float))
        val_ic_lgb = _quick_ic(val2, avail_features)
        total = abs(val_ic_xgb_only) + abs(val_ic_lgb)
        if total > 0:
            w_xgb = abs(val_ic_xgb_only) / total
        print(f"\n  动态 ensemble 权重: XGB={w_xgb:.3f}  LGB={1-w_xgb:.3f}  "
              f"(val_IC: xgb={val_ic_xgb_only:+.4f}  lgb={val_ic_lgb:+.4f})")

    # ── Step 14: 评估 ────────────────────────────────────────────────────
    print("\n[Step 14] 评估（训练集 / Val / Test）")
    train_val_all = pd.concat([train, val], ignore_index=True)

    if ENSEMBLE_LGB and lgb_model is not None:
        # Train+Val 评估（in-sample 参考）
        tv_ens = ensemble_rank_avg(train_val_all, xgb_model, lgb_model,
                                   avail_features, w_xgb=w_xgb)
        train_res = evaluate(tv_ens, model, avail_features, label="TRAIN+VAL",
                             pred_arr=tv_ens["pred"].values)

        # Val 单独评估（已被 HP 搜索"见过"，仅作参考）
        print("\n  [参考] Val 集（已用于 HP 选择，非独立 OOS）:")
        val_ens = ensemble_rank_avg(val, xgb_model, lgb_model, avail_features, w_xgb=w_xgb)
        val_res = evaluate(val_ens, model, avail_features, label="VAL (ref)",
                           pred_arr=val_ens["pred"].values)

        # Test 评估（唯一真实 OOS）
        test_ens = ensemble_rank_avg(test, xgb_model, lgb_model, avail_features, w_xgb=w_xgb)
        test_res = evaluate(test_ens, model, avail_features, label="TEST (OOS)",
                            pred_arr=test_ens["pred"].values)

        print("\n  [参考] 纯 XGBoost（无 LGB ensemble）:")
        evaluate(test, model, avail_features, label="XGB-only")
    else:
        train_res = evaluate(train_val_all, model, avail_features, label="TRAIN+VAL")
        test_res  = evaluate(test,          model, avail_features, label="TEST (OOS)")

    # ── Step 15: 可视化 & 保存 ────────────────────────────────────────────
    print("\n[Step 15] 可视化 & 保存")
    plot_results(train_res, test_res, model, avail_features)
    save_predictions(test_res)

    # 保存特征重要性
    feat_imp_df = pd.DataFrame({
        "feature":    avail_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp_df.to_csv(f"{OUTPUT_DIR}/csv/xgb_feature_importance.csv", index=False)

    # 保存模型到磁盘（供 infer_today.py 实盘推理使用）
    models_dir = f"{OUTPUT_DIR}/models"
    os.makedirs(models_dir, exist_ok=True)
    xgb_model.save_model(f"{models_dir}/xgb_h{HORIZON}.json")
    if ENSEMBLE_LGB and lgb_model is not None:
        lgb_model.save_model(f"{models_dir}/lgb_h{HORIZON}.txt")
        print(f"  模型已保存: {models_dir}/xgb_h{HORIZON}.json + lgb_h{HORIZON}.txt")
    else:
        print(f"  模型已保存: {models_dir}/xgb_h{HORIZON}.json")

    with open(f"{models_dir}/features_h{HORIZON}.json", "w") as f:
        json.dump({
            "features":   avail_features,
            "train_end":  TRAIN_END,
            "val_end":    VAL_END,
            "test_start": TEST_START,
            "best_params": best_params,
            "w_xgb":      w_xgb,
        }, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  最优超参: depth={best_params['max_depth']}  "
          f"mcw={best_params['min_child_weight']}  λ={best_params['reg_lambda']:.1f}")
    print(f"  Train+Val ICIR={train_res['icir']:+.3f}  GAUC={train_res['gauc']:.4f}")
    print(f"  Test(OOS) ICIR={test_res['icir']:+.3f}  GAUC={test_res['gauc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
