#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘推理脚本 ── 输入最新数据，输出当日买卖信号

使用方法:
  python infer_today.py                         # 使用最新可用日期
  python infer_today.py --date 20260314         # 指定日期
  python infer_today.py --holdings hold.json    # 加载当前持仓（进行退出信号检查）
  python infer_today.py --status                # 状态看板（DB状态/MA状态/CS预测/近期交易）

先决条件（按顺序执行一次）:
  python xgboost_cross_section.py               # 训练 H10 模型（保存至 output/models/）
  python xgboost_cross_section_h5.py             # 训练 H5 模型
  python index_timing_model.py --label_type ma60_state --no_wfo  # 生成择时信号

输出:
  1. 大盘择时信号 (slots: 0 / 10 / 20)
  2. 今日买入候选列表（H10×80%+H5×20% ensemble 排名，去除涨停）
  3. 当前持仓退出信号（止损 / MA死叉）
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DB_PATH    = str(ROOT / "data" / "quant.duckdb")
MODELS_DIR = ROOT / "output" / "models"

# 优先使用生产模型（_prod），回退到 eval 模型
def _prod_or(name: str) -> Path:
    """返回 _prod 版本路径（若存在），否则返回原始路径。"""
    p = MODELS_DIR / name
    stem, suffix = name.rsplit(".", 1) if "." in name else (name, "")
    prod = MODELS_DIR / f"{stem}_prod.{suffix}"
    return prod if prod.exists() else p

_timing_prod = ROOT / "output" / "csv" / "index_timing_predictions_prod.csv"
_timing_eval = ROOT / "output" / "csv" / "index_timing_predictions.csv"
INDEX_TIMING_FILE = _timing_prod if _timing_prod.exists() else _timing_eval

# ── 策略参数（与回测保持一致）───────────────────────────────────────────────
MAX_SLOTS       = 8
HALF_SLOTS      = 2           # HP Search v2 最优（原为3）
TOP_K           = 20          # 输出 buy 候选数量
STOP_LOSS_ENTRY = 0.10        # HP Search v2 最优（原为0.08）
MA_DEATH_DAYS   = 3           # 中性市场 MA 死叉阈值（NEUTRAL_MA_DEATH）
MIN_HOLD_DAYS   = 5           # 最短持有期（天内不触发 MA 死叉退出）
MIN_MKTCAP      = 2.0         # 最小市值（亿元）
MKTCAP_PCT_CUT  = 10          # 排除市值最小 10% 分位
MIN_LISTED_DAYS = 90          # 上市不满 N 天排除
REBAL_FREQ      = 5           # 调仓频率（交易日）
WARMUP_DAYS     = 260         # 特征计算所需历史交易日（MA120 + RSI14 + 缓冲）

# ── 状态看板额外路径 ──────────────────────────────────────────────────────────
CS_PRED_FILE  = ROOT / "output" / "csv" / "xgb_cross_section_predictions.csv"
TRADES_FILE   = ROOT / "output" / "index_ma_combined" / "index_ma_combined_trades.csv"
SLOT_CONFIRM_DAYS = 3         # 连续 N 天 slots>0 才确认开新仓
_DIVIDER = '═' * 72

# ─── 特征列（与训练脚本保持一致）─────────────────────────────────────────────
TECH_COLS    = ['ret_1d', 'ret_5d', 'ret_20d', 'ret_60d', 'ret_120d',
                'vol_20d', 'close_vs_ma20', 'close_vs_ma60', 'close_vs_ma120', 'rsi_14',
                'amplitude_1d', 'open_vs_close', 'dist_from_high_5d', 'dist_from_low_5d',
                'dist_from_high_20d', 'high_low_ratio_20d', 'vol_ratio_5_20',
                'vwap_dev_1d', 'vwap_dev_ma5']
BASIC_COLS   = ['pe_ttm', 'pb', 'log_mktcap', 'turnover_20d', 'volume_ratio', 'dv_ratio']
MF_COLS      = ['mf_1d_mv', 'mf_5d_mv', 'mf_20d_mv',
                'large_net_5d_ratio', 'large_net_20d_ratio',
                'retail_net_5d_ratio', 'smart_retail_divergence_5d']
FUND_COLS    = ['roe_ann', 'roa', 'fscore', 'rev_growth_yoy', 'ni_growth_yoy',
                'gross_margin_chg_yoy']
ANALYST_COLS = ['analyst_count', 'np_yield', 'analyst_rev_30d']
HOLDER_COLS  = ['holder_chg_qoq']
CROSS_COLS   = ['smart_momentum', 'momentum_adj_reversal', 'quality_value_score']
SUE_COLS     = ['sue']
ALL_FEATURES = TECH_COLS + BASIC_COLS + MF_COLS + FUND_COLS + ANALYST_COLS + HOLDER_COLS + SUE_COLS + CROSS_COLS


# ════════════════════════════════════════════════════════════════════════════
# 1. 模型加载
# ════════════════════════════════════════════════════════════════════════════

def load_models():
    """加载已保存的 XGB (H10)、LGB (H10)、XGB (H5) 模型"""
    models = {}
    features = {}

    for hz in [10, 5]:
        xgb_path  = _prod_or(f"xgb_h{hz}.json")
        feat_path = _prod_or(f"features_h{hz}.json")
        is_prod   = "_prod" in xgb_path.name
        if not xgb_path.exists():
            print(f"  ⚠ 未找到模型: {xgb_path}")
            print(f"    请先运行: python xgboost_cross_section{'_h5' if hz==5 else ''}.py [--prod]")
            continue
        m = xgb.XGBRegressor()
        m.load_model(str(xgb_path))
        models[f"xgb_h{hz}"] = m
        tag = "[PROD]" if is_prod else "[eval]"
        if feat_path.exists():
            with open(feat_path) as f:
                meta = json.load(f)
            features[f"h{hz}"] = meta.get("features", ALL_FEATURES)
            print(f"  ✓ xgb_h{hz} {tag} 加载（训练截止 {meta.get('train_end', '?')}）")
        else:
            features[f"h{hz}"] = ALL_FEATURES

        lgb_path = _prod_or(f"lgb_h{hz}.txt")
        if lgb_path.exists():
            models[f"lgb_h{hz}"] = lgb.Booster(model_file=str(lgb_path))
            lgb_tag = "[PROD]" if "_prod" in lgb_path.name else "[eval]"
            print(f"  ✓ lgb_h{hz} {lgb_tag} 加载")

    return models, features


# ════════════════════════════════════════════════════════════════════════════
# 2. 确定推理日期 & 数据窗口
# ════════════════════════════════════════════════════════════════════════════

def get_inference_date(conn, target_date: str = None):
    """
    返回 (infer_date, warmup_start, is_rebal_day)
    - infer_date: 推理目标日期（最新可用交易日或指定日期）
    - warmup_start: 数据加载起点（WARMUP_DAYS 个交易日前）
    - is_rebal_day: 该日是否为调仓日
    """
    # 获取所有交易日
    all_td = conn.execute("""
        SELECT DISTINCT trade_date
        FROM daily_price
        WHERE ts_code = '000001.SZ'
        ORDER BY trade_date
    """).fetchdf()["trade_date"].astype(str).tolist()

    if not all_td:
        raise RuntimeError("daily_price 无数据")

    if target_date:
        target_date = target_date.replace("-", "")
        # 找最近的交易日（<=目标日期）
        avail = [d for d in all_td if d <= target_date]
        if not avail:
            raise ValueError(f"日期 {target_date} 之前无交易数据")
        infer_date = avail[-1]
    else:
        infer_date = all_td[-1]

    # 判断是否为调仓日（每5个交易日）
    idx = all_td.index(infer_date)
    is_rebal = (idx % REBAL_FREQ == 0)

    # 最近调仓日（<=infer_date）和下一个调仓日（>infer_date）
    rebal_dates_all = [d for i, d in enumerate(all_td) if i % REBAL_FREQ == 0]
    latest_rebal = max(d for d in rebal_dates_all if d <= infer_date)
    future_rebal = [d for d in rebal_dates_all if d > infer_date]
    next_rebal = future_rebal[0] if future_rebal else None

    # warmup 起点
    warmup_idx = max(0, idx - WARMUP_DAYS - 10)
    warmup_start = all_td[warmup_idx]

    return infer_date, latest_rebal, next_rebal, warmup_start, is_rebal, all_td


# ════════════════════════════════════════════════════════════════════════════
# 3. 特征加载（与训练脚本一致，限制日期范围）
# ════════════════════════════════════════════════════════════════════════════

def load_tech_features(conn, warmup_start: str, infer_date: str) -> pd.DataFrame:
    df = conn.execute(f"""
        WITH dp AS (
            SELECT ts_code, trade_date, open, high, low, close, pct_chg, vol,
                   amount, adj_factor
            FROM daily_price
            WHERE trade_date >= '{warmup_start}' AND trade_date <= '{infer_date}'
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
                -- Alpha158 features
                (high - low) / NULLIF(LAG(close,1) OVER w, 0)  AS amplitude_1d,
                (close - open) / NULLIF(open, 0)               AS open_vs_close,
                close / NULLIF(MAX(high) OVER (w ROWS BETWEEN 4  PRECEDING AND CURRENT ROW), 0) - 1 AS dist_from_high_5d,
                close / NULLIF(MAX(high) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) - 1 AS dist_from_high_20d,
                close / NULLIF(MIN(low)  OVER (w ROWS BETWEEN 4  PRECEDING AND CURRENT ROW), 0) - 1 AS dist_from_low_5d,
                MAX(high) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) /
                  NULLIF(MIN(low) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) - 1
                                                              AS high_low_ratio_20d,
                AVG(vol) OVER (w ROWS BETWEEN 4  PRECEDING AND CURRENT ROW) /
                NULLIF(AVG(vol) OVER (w ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) AS vol_ratio_5_20,
                -- VWAP 偏离度：amount单位千元，vol单位手(100股)；adj_factor抵消
                (close / NULLIF(adj_factor, 0) - amount * 10.0 / NULLIF(vol, 0))
                  / NULLIF(amount * 10.0 / NULLIF(vol, 0), 0)   AS vwap_dev_1d,
                AVG((close / NULLIF(adj_factor, 0) - amount * 10.0 / NULLIF(vol, 0))
                  / NULLIF(amount * 10.0 / NULLIF(vol, 0), 0))
                  OVER (w ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
                                                              AS vwap_dev_ma5
            FROM dp
            WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
        ),
        db AS (
            SELECT
                ts_code, trade_date,
                pe_ttm, pb,
                LN(NULLIF(total_mv, 0))   AS log_mktcap,
                total_mv / 10000.0        AS total_mv_100m,
                AVG(turnover_rate) OVER (
                    PARTITION BY ts_code ORDER BY trade_date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                )                         AS turnover_20d,
                volume_ratio,
                COALESCE(dv_ratio, 0.0)   AS dv_ratio
            FROM daily_basic
            WHERE trade_date >= '{warmup_start}' AND trade_date <= '{infer_date}'
              AND total_mv > 0
        )
        SELECT
            t.ts_code, t.trade_date, t.close,
            t.ret_1d, t.ret_5d, t.ret_20d, t.ret_60d, t.ret_120d, t.vol_20d,
            t.close / NULLIF(t.ma20,   0) - 1            AS close_vs_ma20,
            t.close / NULLIF(t.ma60,   0) - 1            AS close_vs_ma60,
            t.close / NULLIF(t.ma120,  0) - 1            AS close_vs_ma120,
            t.amplitude_1d, t.open_vs_close,
            t.dist_from_high_5d, t.dist_from_low_5d,
            t.dist_from_high_20d, t.high_low_ratio_20d, t.vol_ratio_5_20,
            t.vwap_dev_1d, t.vwap_dev_ma5,
            d.pe_ttm, d.pb, d.log_mktcap, d.total_mv_100m,
            d.turnover_20d, d.volume_ratio, d.dv_ratio
        FROM tech t
        JOIN db d ON t.ts_code = d.ts_code AND t.trade_date = d.trade_date
        WHERE t.ret_60d IS NOT NULL
          AND d.log_mktcap IS NOT NULL
    """).fetchdf()
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def compute_rsi(close_pivot: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = close_pivot.diff(1)
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_moneyflow(conn, warmup_start: str, infer_date: str) -> pd.DataFrame:
    df = conn.execute(f"""
        SELECT
            ts_code, trade_date,
            net_mf_amount AS mf_1d_raw,
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
            SUM(buy_sm_amount - sell_sm_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )  AS retail_net_5d,
            SUM(buy_sm_amount + buy_md_amount + buy_lg_amount + buy_elg_amount
                + sell_sm_amount + sell_md_amount + sell_lg_amount + sell_elg_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            )  AS total_flow_5d,
            SUM(buy_sm_amount + buy_md_amount + buy_lg_amount + buy_elg_amount
                + sell_sm_amount + sell_md_amount + sell_lg_amount + sell_elg_amount) OVER (
                PARTITION BY ts_code ORDER BY trade_date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            )  AS total_flow_20d
        FROM moneyflow
        WHERE trade_date >= '{warmup_start}' AND trade_date <= '{infer_date}'
    """).fetchdf()
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def load_fundamental(conn) -> pd.DataFrame:
    df = conn.execute("""
        WITH is_t AS (
            SELECT ts_code, end_date,
                   MIN(COALESCE(f_ann_date, ann_date)) AS f_ann_date,
                   FIRST(revenue      ORDER BY ann_date DESC) AS revenue,
                   FIRST(oper_cost    ORDER BY ann_date DESC) AS oper_cost,
                   FIRST(n_income_attr_p ORDER BY ann_date DESC) AS n_income,
                   FIRST(basic_eps    ORDER BY ann_date DESC) AS eps
            FROM income_statement WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        ),
        bs_t AS (
            SELECT ts_code, end_date,
                   FIRST(total_assets     ORDER BY ann_date DESC) AS total_assets,
                   FIRST(total_liab       ORDER BY ann_date DESC) AS total_liab,
                   FIRST(total_hldr_eqy_inc_min_int ORDER BY ann_date DESC) AS equity,
                   FIRST(total_cur_assets ORDER BY ann_date DESC) AS cur_assets,
                   FIRST(total_cur_liab   ORDER BY ann_date DESC) AS cur_liab
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
               b.total_assets, b.total_liab, b.equity, b.cur_assets, b.cur_liab, c.ocf
        FROM is_t i
        LEFT JOIN bs_t b USING (ts_code, end_date)
        LEFT JOIN cf_t c USING (ts_code, end_date)
        ORDER BY ts_code, end_date
    """).fetchdf()

    ta  = df["total_assets"].replace(0, np.nan)
    eq  = df["equity"].replace(0, np.nan)
    rev = df["revenue"].replace(0, np.nan)
    cl  = df["cur_liab"].replace(0, np.nan)

    df["period_months"] = df["end_date"].astype(str).str[4:6].map(
        {"03": 3, "06": 6, "09": 9, "12": 12}).fillna(12).astype(int)
    df["pm_mult"]   = 12.0 / df["period_months"]
    df["roe_ann"]   = df["n_income"] / eq * 100.0 * df["pm_mult"]
    df["roa"]       = df["n_income"] / ta * df["pm_mult"] * 100.0
    df["gross_margin"] = (df["revenue"] - df["oper_cost"]) / rev * 100.0
    df["debt_ratio"]   = df["total_liab"] / ta
    df["current_ratio"]= df["cur_assets"] / cl
    df["assets_turn"]  = df["revenue"] / ta

    df["end_date"] = df["end_date"].astype(str)
    dd = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last").copy()
    dd["mmdd"] = dd["end_date"].str[4:]
    dd["prev_end_date"] = (dd["end_date"].str[:4].astype(int) - 1).astype(str) + dd["mmdd"]

    dp = dd[["ts_code", "end_date", "roa", "debt_ratio", "current_ratio",
             "gross_margin", "assets_turn", "n_income", "revenue", "ocf"]].rename(columns={
        "end_date": "prev_end_date", "roa": "roa_p", "debt_ratio": "da_p",
        "current_ratio": "cr_p", "gross_margin": "gpm_p", "assets_turn": "at_p",
        "n_income": "ni_p", "revenue": "rev_p", "ocf": "ocf_p"})
    dd = dd.merge(dp, on=["ts_code", "prev_end_date"], how="left")

    roa_v = dd["roa"].fillna(0.0)
    ocf_v = dd["ocf"].fillna(0.0)
    ta_v  = dd["total_assets"].replace(0, np.nan)
    dd["f1"] = (roa_v > 0).astype(int)
    dd["f2"] = (ocf_v > 0).astype(int)
    dd["f3"] = (dd["n_income"].notna() & dd["ni_p"].notna() & (dd["n_income"] > dd["ni_p"])).astype(int)
    dd["f4"] = (ocf_v / ta_v > roa_v).astype(int)
    dd["f5"] = (dd["debt_ratio"].notna() & dd["da_p"].notna() & (dd["debt_ratio"] < dd["da_p"])).astype(int)
    dd["f6"] = (dd["current_ratio"].notna() & dd["cr_p"].notna() & (dd["current_ratio"] > dd["cr_p"])).astype(int)
    dd["f7"] = (dd["revenue"].notna() & dd["rev_p"].notna() & (dd["revenue"] > dd["rev_p"])).astype(int)
    dd["f8"] = (dd["gross_margin"].notna() & dd["gpm_p"].notna() & (dd["gross_margin"] > dd["gpm_p"])).astype(int)
    dd["f9"] = (dd["assets_turn"].notna() & dd["at_p"].notna() & (dd["assets_turn"] > dd["at_p"])).astype(int)
    dd["fscore"] = dd[["f1","f2","f3","f4","f5","f6","f7","f8","f9"]].sum(axis=1)

    ni_safe = dd["n_income"].replace(0, np.nan)
    dd["rev_growth_yoy"] = np.where(dd["rev_p"].notna() & (dd["rev_p"] != 0),
        (dd["revenue"] - dd["rev_p"]) / dd["rev_p"].abs(), np.nan)
    dd["ni_growth_yoy"]  = np.where(dd["ni_p"].notna() & (dd["ni_p"] > 0),
        (dd["n_income"] - dd["ni_p"]) / dd["ni_p"].abs(), np.nan)
    dd["gross_margin_chg_yoy"] = np.where(dd["gross_margin"].notna() & dd["gpm_p"].notna(),
        dd["gross_margin"] - dd["gpm_p"], np.nan)
    dd["ocf_to_ni"] = np.where(ni_safe.notna() & dd["ocf"].notna(),
        dd["ocf"] / ni_safe.abs(), np.nan)

    keep = ["ts_code", "end_date", "fscore", "rev_growth_yoy", "ni_growth_yoy",
            "gross_margin_chg_yoy", "ocf_to_ni"]
    df = df.merge(dd[keep], on=["ts_code", "end_date"], how="left")

    fund_cols = ["ts_code", "f_ann_date", "end_date",
                 "roe_ann", "roa", "fscore", "rev_growth_yoy", "ni_growth_yoy",
                 "gross_margin_chg_yoy"]
    result = df[fund_cols].dropna(subset=["f_ann_date"]).copy()
    result["f_ann_date"] = result["f_ann_date"].astype(str)
    return result.sort_values(["ts_code", "f_ann_date"]).reset_index(drop=True)


def pit_join(fund_df: pd.DataFrame, ts_codes: list, infer_date: str) -> pd.DataFrame:
    """对给定股票列表，PIT 获取 infer_date 前最新已披露的基本面数据"""
    keys = pd.DataFrame({"ts_code": ts_codes, "trade_date": infer_date})
    keys["_td_int"] = int(infer_date.replace("-", ""))
    fund_df = fund_df.copy()
    fund_df["_ann_int"] = fund_df["f_ann_date"].str.replace("-", "").astype(int)

    fund_cols = ["roe_ann", "roa", "fscore", "rev_growth_yoy", "ni_growth_yoy",
                 "gross_margin_chg_yoy"]

    grouped = {ts: grp.sort_values("_ann_int").reset_index(drop=True)
               for ts, grp in fund_df.groupby("ts_code")}

    results = []
    for _, row in keys.iterrows():
        ts = row["ts_code"]
        td_int = row["_td_int"]
        rec = {"ts_code": ts, "trade_date": infer_date}
        if ts in grouped:
            grp = grouped[ts]
            ann = grp["_ann_int"].values
            idx = np.searchsorted(ann, td_int, side="right") - 1
            if idx >= 0:
                for col in fund_cols:
                    if col in grp.columns:
                        rec[col] = grp[col].iloc[idx]
        results.append(rec)

    return pd.DataFrame(results)


def load_analyst(conn, infer_date: str, mv_map: dict) -> pd.DataFrame:
    win_start = (pd.to_datetime(infer_date) - pd.Timedelta(days=90)).strftime("%Y%m%d")
    old_start = (pd.to_datetime(infer_date) - pd.Timedelta(days=90)).strftime("%Y%m%d")
    rc_df = conn.execute(f"""
        SELECT ts_code, report_date, np
        FROM report_rc
        WHERE np IS NOT NULL AND np > 0
          AND report_date >= '{old_start}' AND report_date <= '{infer_date}'
        ORDER BY ts_code, report_date
    """).fetchdf()
    rc_df["report_date"] = rc_df["report_date"].astype(str)

    if rc_df.empty:
        return pd.DataFrame(columns=["ts_code", "analyst_count", "np_yield", "analyst_rev_30d"])

    # analyst_count and np_yield (last 90d)
    agg = rc_df.groupby("ts_code")["np"].agg(analyst_count="count", np_median="median").reset_index()
    agg["np_yield"] = agg.apply(
        lambda r: r["np_median"] / mv_map.get(r["ts_code"], np.nan)
        if pd.notna(mv_map.get(r["ts_code"], np.nan)) and mv_map.get(r["ts_code"], 0) > 0
        else np.nan, axis=1)
    agg = agg.drop(columns=["np_median"])

    # analyst_rev_30d: (recent 30d median) / (31-90d median) - 1
    recent_start = (pd.to_datetime(infer_date) - pd.Timedelta(days=30)).strftime("%Y%m%d")
    old_end      = (pd.to_datetime(infer_date) - pd.Timedelta(days=31)).strftime("%Y%m%d")
    mask_recent = (rc_df["report_date"] >= recent_start) & (rc_df["report_date"] <= infer_date)
    mask_old    = (rc_df["report_date"] >= old_start) & (rc_df["report_date"] <= old_end)
    recent_np = rc_df[mask_recent].groupby("ts_code")["np"].median().rename("recent_np")
    old_np    = rc_df[mask_old   ].groupby("ts_code")["np"].median().rename("old_np")
    rev = pd.concat([recent_np, old_np], axis=1).reset_index()
    rev["analyst_rev_30d"] = (rev["recent_np"] / rev["old_np"].replace(0, np.nan) - 1).clip(-1.0, 2.0)
    agg = agg.merge(rev[["ts_code", "analyst_rev_30d"]], on="ts_code", how="left")
    return agg


def load_holder_pit(conn, ts_codes: list, infer_date: str) -> pd.DataFrame:
    try:
        hdf = conn.execute("""
            SELECT ts_code, COALESCE(ann_date, end_date) AS f_ann_date, end_date, holder_num
            FROM stk_holdernumber WHERE holder_num > 0
            ORDER BY ts_code, end_date
        """).fetchdf()
    except Exception:
        return pd.DataFrame({"ts_code": ts_codes, "holder_chg_qoq": np.nan})

    if hdf.empty:
        return pd.DataFrame({"ts_code": ts_codes, "holder_chg_qoq": np.nan})

    hdf["f_ann_date"] = hdf["f_ann_date"].astype(str)
    hdf = hdf.sort_values(["ts_code", "end_date"])
    hdf["prev_num"] = hdf.groupby("ts_code")["holder_num"].shift(1)
    hdf["holder_chg_qoq"] = np.where(
        hdf["prev_num"].notna() & (hdf["prev_num"] > 0),
        (hdf["holder_num"] - hdf["prev_num"]) / hdf["prev_num"], np.nan)
    hdf = hdf.dropna(subset=["holder_chg_qoq"])
    hdf["_ann_int"] = hdf["f_ann_date"].str.replace("-", "").astype(int)
    td_int = int(infer_date.replace("-", ""))

    results = []
    for ts in ts_codes:
        grp = hdf[hdf["ts_code"] == ts].sort_values("_ann_int")
        if grp.empty:
            results.append({"ts_code": ts, "holder_chg_qoq": np.nan})
        else:
            ann = grp["_ann_int"].values
            idx = np.searchsorted(ann, td_int, side="right") - 1
            if idx >= 0:
                results.append({"ts_code": ts, "holder_chg_qoq": grp["holder_chg_qoq"].iloc[idx]})
            else:
                results.append({"ts_code": ts, "holder_chg_qoq": np.nan})
    return pd.DataFrame(results)


def load_sue_pit(conn, infer_date: str, max_lag_days: int = 90) -> pd.DataFrame:
    """PIT SUE: (actual_np - consensus_np) / |consensus_np|, 披露后 max_lag_days 内有效"""
    is_df = conn.execute("""
        WITH is_t AS (
            SELECT ts_code, end_date,
                   MIN(COALESCE(f_ann_date, ann_date)) AS f_ann_date,
                   FIRST(n_income_attr_p ORDER BY ann_date DESC) AS actual_np
            FROM income_statement WHERE comp_type = '1'
            GROUP BY ts_code, end_date
        )
        SELECT * FROM is_t WHERE f_ann_date IS NOT NULL AND actual_np IS NOT NULL
    """).fetchdf()
    if is_df.empty:
        return pd.DataFrame(columns=["ts_code", "sue"])

    month_to_q = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "Q4"}
    is_df["quarter"] = is_df["end_date"].astype(str).str[:4] + \
                       is_df["end_date"].astype(str).str[4:6].map(month_to_q)
    is_df["f_ann_date"] = is_df["f_ann_date"].astype(str)

    rc_df = conn.execute("""
        SELECT ts_code, report_date, quarter, np FROM report_rc
        WHERE np IS NOT NULL AND np > 0
    """).fetchdf()
    rc_df["report_date"] = rc_df["report_date"].astype(str)

    merged = is_df.merge(rc_df, on=["ts_code", "quarter"])
    if merged.empty:
        return pd.DataFrame(columns=["ts_code", "sue"])

    days_before = (pd.to_datetime(merged["f_ann_date"]) - pd.to_datetime(merged["report_date"])).dt.days
    merged = merged[(days_before >= 5) & (days_before <= 180)]
    consensus = merged.groupby(["ts_code", "end_date", "f_ann_date", "actual_np"])["np"].agg(
        consensus_median="median", n_analysts="count").reset_index()
    consensus = consensus[consensus["n_analysts"] >= 2]
    consensus["sue"] = ((consensus["actual_np"] - consensus["consensus_median"]) /
                        consensus["consensus_median"].abs().replace(0, np.nan)).clip(-2.0, 2.0)

    # PIT: keep only f_ann_date <= infer_date and within max_lag_days
    td_dt = pd.to_datetime(infer_date)
    sue_valid = consensus[pd.to_datetime(consensus["f_ann_date"]) <= td_dt].copy()
    sue_valid["days_since"] = (td_dt - pd.to_datetime(sue_valid["f_ann_date"])).dt.days
    sue_valid = sue_valid[sue_valid["days_since"] <= max_lag_days]
    # For each stock, keep the most recent announcement
    sue_valid = sue_valid.sort_values("f_ann_date").groupby("ts_code").last().reset_index()
    return sue_valid[["ts_code", "sue"]]


# ════════════════════════════════════════════════════════════════════════════
# 4. 预处理（与训练脚本完全一致）
# ════════════════════════════════════════════════════════════════════════════

def winsorize_mad(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    median = series.median()
    if pd.isna(median):
        return series
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + threshold * 1.4826 * mad
    lower = median - threshold * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)


def preprocess_single_date(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """对单一截面做 MAD → 中性化 → Z-score（与训练时 process_group 完全一致）"""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = winsorize_mad(df[col])

    if len(df) > 20 and "industry" in df.columns and "log_mktcap" in df.columns:
        neut_cols = [c for c in feature_cols
                     if c not in ("log_mktcap", "fscore", "analyst_count",
                                  "analyst_rev_30d", "sue")]
        industries = pd.get_dummies(df["industry"], prefix="ind", drop_first=True, dtype=float)
        logmv = df["log_mktcap"].fillna(df["log_mktcap"].median())
        X_ctrl = pd.concat([logmv.rename("log_mktcap"), industries], axis=1).values.astype(float)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        for col in neut_cols:
            if col not in df.columns:
                continue
            y = df[col].values.astype(float)
            valid = ~np.isnan(y)
            if valid.sum() < 10:
                continue
            ridge.fit(X_ctrl[valid], y[valid])
            pred = ridge.predict(X_ctrl)
            df[col] = np.where(valid, y - pred + pred[valid].mean(), np.nan)

    for col in feature_cols:
        if col in df.columns:
            s = df[col]
            mu, std = s.mean(), s.std()
            if std > 0:
                df[col] = (s - mu) / std
            else:
                df[col] = 0.0

    return df


# ════════════════════════════════════════════════════════════════════════════
# 5. 指数择时信号
# ════════════════════════════════════════════════════════════════════════════

def get_timing_slots(infer_date: str):
    """从已生成的 CSV 读取指数择时 slots"""
    if not INDEX_TIMING_FILE.exists():
        print(f"  ⚠ 择时文件不存在: {INDEX_TIMING_FILE}")
        print(f"    请先运行: python index_timing_model.py --label_type ma60_state --no_wfo [--prod]")
        return None, "未知（缺少择时文件）"

    df = pd.read_csv(INDEX_TIMING_FILE, dtype={"trade_date": str})
    df["trade_date"] = df["trade_date"].str.replace("-", "")
    df = df.sort_values("trade_date")

    avail = df[df["trade_date"] <= infer_date]
    if avail.empty:
        return None, "未知（数据不足）"

    row = avail.iloc[-1]
    slots = int(row["slots"])
    state = {0: "熊市/空仓 (slots=0)", 10: "半仓 (slots=10)", 20: "满仓 (slots=20)"}
    return slots, state.get(slots, f"slots={slots}")


# ════════════════════════════════════════════════════════════════════════════
# 6. 股票合规性过滤
# ════════════════════════════════════════════════════════════════════════════

def get_eligible_stocks(conn, infer_date: str, all_td: list) -> set:
    """返回当日合规股（市值、上市天数、非ST）"""
    df = conn.execute(f"""
        SELECT sb.ts_code, sb.list_date,
               db.total_mv / 10000.0 AS mktcap_100m
        FROM stock_basic sb
        LEFT JOIN daily_basic db
            ON sb.ts_code = db.ts_code AND db.trade_date = '{infer_date}'
        WHERE sb.ts_code NOT LIKE '8%'
          AND sb.ts_code NOT LIKE '4%'
          AND sb.name NOT LIKE '%ST%'
    """).fetchdf()

    # 上市天数过滤
    df["list_date"] = df["list_date"].astype(str)
    td_idx = {d: i for i, d in enumerate(all_td)}
    infer_idx = td_idx.get(infer_date, len(all_td))
    def listed_days(ld):
        for i, d in enumerate(all_td):
            if d >= ld:
                return infer_idx - i
        return infer_idx
    df["days_listed"] = df["list_date"].apply(listed_days)
    df = df[df["days_listed"] >= MIN_LISTED_DAYS]

    # 市值过滤
    df = df.dropna(subset=["mktcap_100m"])
    df = df[df["mktcap_100m"] >= MIN_MKTCAP]
    pct_cut = df["mktcap_100m"].quantile(MKTCAP_PCT_CUT / 100)
    df = df[df["mktcap_100m"] >= pct_cut]

    return set(df["ts_code"])


# ════════════════════════════════════════════════════════════════════════════
# 7. 生成预测
# ════════════════════════════════════════════════════════════════════════════

def predict_scores(models: dict, features_map: dict, panel_today: pd.DataFrame) -> pd.DataFrame:
    """
    生成 H10+H5 ensemble 得分（与回测策略 load_cs_predictions 逻辑一致）。
    H10: XGB + LGB rank 平均 → r10
    H5:  XGB only            → r5
    final pred = 0.8 * r10 + 0.2 * r5
    """
    df = panel_today.copy()

    # H10
    if "xgb_h10" in models:
        feats_h10 = features_map.get("h10", ALL_FEATURES)
        avail = [c for c in feats_h10 if c in df.columns]
        X10 = df[avail].values.astype(float)
        pred_xgb10 = models["xgb_h10"].predict(X10)
        if "lgb_h10" in models:
            pred_lgb10 = models["lgb_h10"].predict(X10)
            # 截面 rank 平均
            r_xgb = pd.Series(pred_xgb10).rank(pct=True).values
            r_lgb = pd.Series(pred_lgb10).rank(pct=True).values
            pred_h10 = 0.5 * r_xgb + 0.5 * r_lgb
        else:
            pred_h10 = pred_xgb10
        df["pred_h10"] = pred_h10
        df["r10"] = df["pred_h10"].rank(pct=True)
    else:
        df["r10"] = 0.5

    # H5
    if "xgb_h5" in models:
        feats_h5 = features_map.get("h5", ALL_FEATURES)
        avail5 = [c for c in feats_h5 if c in df.columns]
        X5 = df[avail5].values.astype(float)
        pred_h5 = models["xgb_h5"].predict(X5)
        df["r5"] = pd.Series(pred_h5).rank(pct=True).values
    else:
        df["r5"] = df.get("r10", 0.5)

    df["score"] = 0.8 * df["r10"] + 0.2 * df["r5"]
    return df


# ════════════════════════════════════════════════════════════════════════════
# 8. 当前持仓退出信号
# ════════════════════════════════════════════════════════════════════════════

def check_exits(holdings: dict, conn, infer_date: str, all_td: list) -> list:
    """
    对当前持仓检查退出信号：
    - 硬止损: 当前价 < 入场价 × (1 - STOP_LOSS_ENTRY)
    - MA死叉: MA5 < MA20 连续 MA_DEATH_DAYS 天（且持有 >= MIN_HOLD_DAYS）
    返回: list of dict { ts_code, reason, current_price, entry_price, pnl_pct }
    """
    if not holdings:
        return []

    ts_list = list(holdings.keys())
    ts_str = "', '".join(ts_list)

    # 加载最近 30 天价格（用于 MA5/MA20 和止损判断）
    recent_idx = max(0, all_td.index(infer_date) - 30)
    start_30 = all_td[recent_idx]

    price_df = conn.execute(f"""
        SELECT ts_code, trade_date, close
        FROM daily_price
        WHERE ts_code IN ('{ts_str}')
          AND trade_date >= '{start_30}'
          AND trade_date <= '{infer_date}'
        ORDER BY ts_code, trade_date
    """).fetchdf()
    price_df["trade_date"] = price_df["trade_date"].astype(str)

    exits = []
    for ts, info in holdings.items():
        grp = price_df[price_df["ts_code"] == ts].sort_values("trade_date")
        if grp.empty:
            continue
        closes = grp["close"].values
        cur_price = closes[-1]
        entry_price = float(info.get("entry_price", cur_price))
        entry_date  = str(info.get("entry_date", ""))
        hold_days   = sum(1 for d in all_td if entry_date <= d <= infer_date) if entry_date else 99

        pnl_pct = (cur_price - entry_price) / entry_price * 100

        # 硬止损
        if cur_price < entry_price * (1 - STOP_LOSS_ENTRY):
            exits.append({
                "ts_code": ts, "reason": f"硬止损 ({STOP_LOSS_ENTRY:.0%})",
                "current_price": cur_price, "entry_price": entry_price,
                "pnl_pct": pnl_pct, "hold_days": hold_days
            })
            continue

        # MA 死叉（只在持满 MIN_HOLD_DAYS 后检查）
        if len(closes) >= 20 and hold_days >= MIN_HOLD_DAYS:
            ma5  = pd.Series(closes).rolling(5,  min_periods=3).mean().values
            ma20 = pd.Series(closes).rolling(20, min_periods=10).mean().values
            # 检查最后 MA_DEATH_DAYS 天是否连续死叉
            recent_n = min(MA_DEATH_DAYS, len(ma5))
            if all(ma5[-(recent_n - i)] < ma20[-(recent_n - i)]
                   for i in range(recent_n) if not np.isnan(ma5[-(recent_n - i)])):
                exits.append({
                    "ts_code": ts, "reason": f"MA死叉({MA_DEATH_DAYS}日确认)",
                    "current_price": cur_price, "entry_price": entry_price,
                    "pnl_pct": pnl_pct, "hold_days": hold_days
                })

    return exits


# ════════════════════════════════════════════════════════════════════════════
# 9. 状态看板（--status 模式，原 daily_inference.py）
# ════════════════════════════════════════════════════════════════════════════

def _st_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def _st_fmt_pct(v, decimals=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  N/A  '
    sign = '+' if v >= 0 else ''
    return f"{sign}{v * 100:.{decimals}f}%"


def _st_bar(val: float, lo: float = 0.0, hi: float = 1.0, width: int = 20) -> str:
    t = np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)
    filled = int(t * width)
    return '█' * filled + '░' * (width - filled)


def _st_load_csi300_ma() -> pd.DataFrame:
    with _st_conn() as conn:
        df = conn.execute("""
            SELECT trade_date, close FROM index_daily
            WHERE ts_code = '000300.SH' ORDER BY trade_date
        """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')
    df = df.set_index('trade_date').sort_index()
    df['ma20']  = df['close'].rolling(20,  min_periods=5).mean()
    df['ma60']  = df['close'].rolling(60,  min_periods=20).mean()
    df['ma250'] = df['close'].rolling(250, min_periods=60).mean()
    df['ret_1d']  = df['close'].pct_change(1)
    df['ret_5d']  = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    return df


def _st_ma_state(row):
    c, ma20, ma60 = row['close'], row['ma20'], row['ma60']
    if pd.isna(ma20) or pd.isna(ma60):
        return 'unknown', 10
    if c < ma20:
        return '熊市 (close < MA20)', 0
    if c < ma60:
        return '中性 (MA20 ≤ close < MA60)', 10
    return '牛市 (close ≥ MA60)', 20


def _st_section_db_status():
    print(f"\n{_DIVIDER}")
    print("  [1] 数据库状态")
    print(_DIVIDER)
    tables = {
        'daily_price':      "SELECT MAX(trade_date) FROM daily_price",
        'index_daily':      "SELECT MAX(trade_date) FROM index_daily",
        'daily_basic':      "SELECT MAX(trade_date) FROM daily_basic",
        'moneyflow':        "SELECT MAX(trade_date) FROM moneyflow",
        'income_statement': "SELECT MAX(ann_date) FROM income_statement",
        'fina_indicator':   "SELECT MAX(ann_date) FROM fina_indicator",
    }
    with _st_conn() as conn:
        for tbl, sql in tables.items():
            try:
                latest = conn.execute(sql).fetchone()[0]
                print(f"  {tbl:<22}  最新: {latest}")
            except Exception as e:
                print(f"  {tbl:<22}  查询失败: {e}")
    print()


def _st_section_timing(display_days: int = 20):
    print(f"\n{_DIVIDER}")
    print("  [2] 大盘择时模型 — CSI300 MA 状态 + 时序模型信号")
    print(_DIVIDER)

    ma_df = _st_load_csi300_ma()
    latest_date = ma_df.index[-1]
    latest = ma_df.loc[latest_date]
    state_label, raw_slots = _st_ma_state(latest)

    print(f"\n  CSI300 最新数据日期: {latest_date}")
    print(f"  {'收盘价':10}  {latest['close']:.2f}")
    print(f"  {'MA20':10}  {latest['ma20']:.2f}   偏离 {_st_fmt_pct(latest['close']/latest['ma20']-1)}")
    print(f"  {'MA60':10}  {latest['ma60']:.2f}   偏离 {_st_fmt_pct(latest['close']/latest['ma60']-1)}")
    print(f"  {'MA250':10}  {latest['ma250']:.2f}   偏离 {_st_fmt_pct(latest['close']/latest['ma250']-1)}")
    print(f"\n  ▶ MA 状态: {state_label}  →  基础 slots = {raw_slots}")
    print(f"  ▶ 1日涨跌: {_st_fmt_pct(latest['ret_1d'])}   "
          f"5日: {_st_fmt_pct(latest['ret_5d'])}   20日: {_st_fmt_pct(latest['ret_20d'])}")

    recent = ma_df.tail(display_days)
    print(f"\n  近 {display_days} 个交易日 CSI300 MA 状态:")
    print(f"  {'日期':10}  {'收盘':>8}  {'MA20':>8}  {'MA60':>8}  {'状态':>6}  {'Slots':>6}")
    for dt, row in recent.iterrows():
        sl, ss = _st_ma_state(row)
        symbol = '▲' if ss == 20 else ('─' if ss == 10 else '▼')
        print(f"  {dt}  {row['close']:8.2f}  {row['ma20']:8.2f}  "
              f"{row['ma60']:8.2f}  {symbol:>6}  {ss:>6}")

    recent_states = [_st_ma_state(r)[1] for _, r in recent.tail(SLOT_CONFIRM_DAYS).iterrows()]
    consec_bull = all(s > 0 for s in recent_states)
    print(f"\n  SLOT_CONFIRM_DAYS={SLOT_CONFIRM_DAYS}: 最近{SLOT_CONFIRM_DAYS}天 "
          f"slots={recent_states}  → {'✓ 可开新仓' if consec_bull else '✗ 等待确认，暂不开新仓'}")

    if INDEX_TIMING_FILE.exists():
        timing_df = pd.read_csv(INDEX_TIMING_FILE, dtype={'trade_date': str})
        timing_df['trade_date'] = timing_df['trade_date'].str.replace('-', '')
        timing_df = timing_df.sort_values('trade_date')
        last_timing_date = timing_df['trade_date'].iloc[-1]
        print(f"\n  时序模型预测（CSV）最新日期: {last_timing_date}")
        print(f"  近 {display_days} 天 pred_prob & slots:")
        print(f"  {'日期':10}  {'pred_prob':>10}  {'slots':>6}  {'概率条'}")
        for _, row in timing_df.tail(display_days).iterrows():
            prob = row['pred_prob']
            sl = int(row['slots'])
            print(f"  {row['trade_date']}  {prob:10.4f}  {sl:6d}  {_st_bar(prob, 0.3, 0.9, 24)}")
        last_30 = timing_df.tail(30)
        print(f"\n  近30天槽位分布: "
              f"0槽={(last_30['slots']==0).sum()}天  "
              f"10槽={(last_30['slots']==10).sum()}天  "
              f"20槽={(last_30['slots']==20).sum()}天")
    else:
        print(f"\n  [!] 未找到时序模型预测文件: {INDEX_TIMING_FILE}")
        print("      请先运行: python index_timing_model.py --label_type ma60_state --no_wfo")

    return raw_slots, consec_bull


def _st_section_cs_model(target_date=None, top_n: int = 20):
    print(f"\n{_DIVIDER}")
    print("  [3] 截面选股模型 — 最新截面日 Top 股票")
    print(_DIVIDER)

    if not CS_PRED_FILE.exists():
        print(f"  [!] 未找到截面选股预测: {CS_PRED_FILE}")
        print("      请先运行: python xgboost_cross_section.py")
        return None, None

    cs = pd.read_csv(CS_PRED_FILE, dtype={'trade_date': str})
    cs['trade_date'] = cs['trade_date'].str.replace('-', '')
    all_dates = sorted(cs['trade_date'].unique())

    if target_date:
        avail = [d for d in all_dates if d <= target_date.replace('-', '')]
        cs_date = avail[-1] if avail else all_dates[-1]
    else:
        cs_date = all_dates[-1]

    print(f"\n  截面预测最新日期: {cs_date}  (总截面数: {len(all_dates)})")
    print(f"  预测文件覆盖范围: {all_dates[0]} ~ {all_dates[-1]}")

    day_cs = cs[cs['trade_date'] == cs_date].copy()
    print(f"  本截面股票数: {len(day_cs):,}")
    print(f"  pred 分布:  min={day_cs['pred'].min():.4f}  "
          f"median={day_cs['pred'].median():.4f}  max={day_cs['pred'].max():.4f}")

    with _st_conn() as conn:
        # 简单合规过滤：排除 ST、北交所
        sb = conn.execute("SELECT ts_code, name FROM stock_basic").fetchdf()
        st_set = set(sb.loc[sb['name'].str.contains('ST', na=False), 'ts_code'])
        mktcap_df = conn.execute(f"""
            SELECT ts_code, total_mv FROM daily_basic
            WHERE trade_date = '{cs_date}' AND total_mv > 0
        """).fetchdf()
        meta = conn.execute("SELECT ts_code, name, industry FROM stock_basic").fetchdf()
        meta = meta.set_index('ts_code')
        latest_price_date = conn.execute(
            "SELECT MAX(trade_date) FROM daily_price WHERE ts_code NOT LIKE '8%'"
        ).fetchone()[0]
        latest_price_date = str(latest_price_date).replace('-', '')

        if not mktcap_df.empty:
            cutoff = np.percentile(mktcap_df['total_mv'].values, 10)
            eligible = set(mktcap_df.loc[mktcap_df['total_mv'] > cutoff, 'ts_code']) - st_set
            eligible = {c for c in eligible if not c.startswith('8') and not c.startswith('4')}
        else:
            eligible = set()

    day_cs_elig = day_cs[day_cs['ts_code'].isin(eligible)].sort_values('pred', ascending=False).reset_index(drop=True)
    print(f"\n  合规过滤后: {len(day_cs_elig):,} 只")

    top_codes = day_cs_elig['ts_code'].head(top_n).tolist()
    with _st_conn() as conn:
        start_dt = (pd.Timestamp(latest_price_date) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        price_df = conn.execute(f"""
            SELECT ts_code, trade_date, close, pct_chg,
                AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma5,
                AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20
            FROM daily_price
            WHERE ts_code IN ({','.join(f"'{c}'" for c in top_codes)})
              AND trade_date >= '{start_dt}' AND trade_date <= '{latest_price_date}'
        """).fetchdf()
        price_df['trade_date'] = price_df['trade_date'].astype(str).str.replace('-', '')
        prices = price_df.sort_values('trade_date').groupby('ts_code').last()

    print(f"\n  Top-{top_n} 截面预测（价格更新至 {latest_price_date}）:")
    print(f"  {'#':>3}  {'代码':12}  {'名称':10}  {'行业':10}  {'pred':>8}  {'分位':>6}  {'当前价':>8}  {'1日':>8}  MA")
    for i, (_, row) in enumerate(day_cs_elig.head(top_n).iterrows(), 1):
        ts = row['ts_code']
        pred = row['pred']
        pct = (day_cs['pred'] < pred).mean() * 100
        name     = str(meta.loc[ts, 'name'])[:8]     if ts in meta.index else '—'
        industry = str(meta.loc[ts, 'industry'])[:8] if ts in meta.index and not isinstance(meta.loc[ts, 'industry'], float) else '—'
        if ts in prices.index:
            pr = prices.loc[ts]
            chg = pr['pct_chg'] / 100 if pd.notna(pr['pct_chg']) else np.nan
            ma5, ma20 = pr['ma5'], pr['ma20']
            ma_sym = '▲' if (pd.notna(ma5) and pd.notna(ma20) and ma5 > ma20) else '▼'
            price_str = f"{pr['close']:8.2f}"
            chg_str   = _st_fmt_pct(chg)
        else:
            price_str, chg_str, ma_sym = '    N/A ', '   N/A  ', '?'
        print(f"  {i:>3}  {ts:12}  {name:10}  {industry:10}  {pred:8.4f}  {pct:5.1f}%  {price_str}  {chg_str:>8}  {ma_sym}")

    return day_cs_elig, cs_date


def _st_section_recommendation(raw_slots: int, consec_bull: bool,
                                cs_df, cs_date):
    print(f"\n{_DIVIDER}")
    print("  [4] 联合策略当前建议")
    print(_DIVIDER)

    if raw_slots == 0:
        effective_slots = 0
        reason = "熊市（CSI300 < MA20）→ 停止开新仓"
    elif not consec_bull:
        effective_slots = 0
        reason = f"bull 信号存在，但不足 {SLOT_CONFIRM_DAYS} 天连续确认 → 暂不开新仓"
    else:
        effective_slots = MAX_SLOTS if raw_slots == 20 else HALF_SLOTS
        reason = f"MA 状态正常 + 连续确认 → {effective_slots} 槽"

    print(f"\n  市场状态 slots (MA规则): {raw_slots}")
    print(f"  SLOT_CONFIRM_DAYS 确认: {'通过 ✓' if consec_bull else '未通过 ✗'}")
    print(f"  ▶ 有效 slots: {effective_slots}")
    print(f"  ▶ 原因: {reason}")

    if cs_df is None or cs_df.empty or effective_slots == 0:
        print(f"\n  ▶ 建议: 不开新仓，现有持仓由风控自然退出")
        return

    print(f"\n  基于截面选股 ({cs_date}) Top-{effective_slots} 建议持仓:")
    print(f"\n  {'#':>3}  {'代码':12}  {'pred':>8}")
    with _st_conn() as conn:
        meta = conn.execute("SELECT ts_code, name FROM stock_basic").fetchdf().set_index('ts_code')
    for i, (_, row) in enumerate(cs_df.head(effective_slots).iterrows(), 1):
        ts = row['ts_code']
        name = str(meta.loc[ts, 'name'])[:10] if ts in meta.index else '—'
        print(f"  {i:>3}  {ts:12}  {name:10}  {row['pred']:8.4f}")


def _st_section_recent_trades(n: int = 15):
    print(f"\n{_DIVIDER}")
    print("  [5] 最近成交记录 (index_ma_combined_strategy 回测)")
    print(_DIVIDER)

    if not TRADES_FILE.exists():
        print(f"  [!] 未找到交易记录: {TRADES_FILE}")
        return

    trades = pd.read_csv(TRADES_FILE, dtype={'date': str, 'ts_code': str})
    real_trades = trades[trades['ts_code'] != 'REBAL'].copy()
    rebal_rows  = trades[trades['ts_code'] == 'REBAL'].copy()
    print(f"\n  总交易记录: {len(real_trades)} 笔  调仓日: {len(rebal_rows)} 次")

    if len(rebal_rows) > 0:
        last_rebal = rebal_rows.iloc[-1]
        print(f"  最近调仓日: {last_rebal['date']}  "
              f"slots={int(last_rebal['price'])}  持仓数={int(last_rebal['cash'])}")

    with _st_conn() as conn:
        meta = conn.execute("SELECT ts_code, name FROM stock_basic").fetchdf().set_index('ts_code')

    print(f"\n  最近 {n} 笔个股交易:")
    print(f"  {'日期':10}  {'代码':12}  {'操作':12}  {'价格':>10}  {'现金余额':>14}")
    for _, row in real_trades.tail(n).iterrows():
        ts = row['ts_code']
        name = str(meta.loc[ts, 'name'])[:6] if ts in meta.index else '—'
        print(f"  {row['date']}  {ts:12}  ({name}) {row['action']:10}  "
              f"{row['price']:>10.2f}  {row['cash']:>14,.2f}")


def _st_section_freshness():
    print(f"\n{_DIVIDER}")
    print("  [6] 数据新鲜度")
    print(_DIVIDER)

    with _st_conn() as conn:
        db_price = str(conn.execute(
            "SELECT MAX(trade_date) FROM daily_price WHERE ts_code NOT LIKE '8%'"
        ).fetchone()[0]).replace('-', '')
        db_index = str(conn.execute(
            "SELECT MAX(trade_date) FROM index_daily"
        ).fetchone()[0]).replace('-', '')

    rows = [('DB daily_price', db_price, '个股价格'), ('DB index_daily', db_index, 'CSI300 MA状态')]
    if INDEX_TIMING_FILE.exists():
        t = pd.read_csv(INDEX_TIMING_FILE, dtype={'trade_date': str})
        rows.append(('时序模型预测', t['trade_date'].str.replace('-', '').max(), 'pred_prob/slots'))
    if CS_PRED_FILE.exists():
        c = pd.read_csv(CS_PRED_FILE, dtype={'trade_date': str})
        rows.append(('截面选股预测', c['trade_date'].str.replace('-', '').max(), 'CS pred score'))

    print(f"\n  {'数据源':16}  {'最新日期':12}  {'用途':20}  {'状态'}")
    for name, date, use in rows:
        try:
            delta = (pd.Timestamp(db_price) - pd.Timestamp(date)).days
            status = f"落后 {delta} 天" if delta > 0 else "最新"
        except Exception:
            status = '—'
        print(f"  {name:16}  {date:12}  {use:20}  {status}")

    print(f"""
  如需刷新预测（约需 5~30 分钟）:
    python index_timing_model.py --label_type ma60_state --no_wfo  # ~5分钟
    python xgboost_cross_section.py                                 # ~25分钟
""")


def show_status(top_n: int = 20, date: str = None):
    """打印当日策略状态看板（原 daily_inference.py 功能）"""
    import datetime as _dt
    print(f"\n{_DIVIDER}")
    print("  每日状态看板  —  指数择时 + 截面选股联合策略")
    print(_DIVIDER)
    print(f"  运行时间: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    _st_section_db_status()
    raw_slots, consec_bull = _st_section_timing(display_days=20)
    cs_df, cs_date = _st_section_cs_model(target_date=date, top_n=top_n)
    _st_section_recommendation(raw_slots, consec_bull, cs_df, cs_date)
    _st_section_recent_trades(n=15)
    _st_section_freshness()
    print(f"\n{_DIVIDER}\n")


# ════════════════════════════════════════════════════════════════════════════
# 10. 主函数
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="实盘推理：输入最新数据，输出买卖信号")
    parser.add_argument("--date",     default=None, help="指定推理日期 YYYYMMDD（默认：最新可用日期）")
    parser.add_argument("--holdings", default=None, help="当前持仓 JSON 文件路径")
    parser.add_argument("--top_k",   type=int, default=TOP_K, help="输出 buy 候选数量")
    parser.add_argument("--status",  action="store_true",
                        help="打印状态看板（DB新鲜度、MA状态、CS预测、最近交易），不运行推理模型")
    args = parser.parse_args()

    if args.status:
        show_status(top_n=args.top_k, date=args.date)
        return

    print("=" * 65)
    print("  实盘推理  |  指数择时 + 截面选股联合策略")
    print("=" * 65)

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    print("\n[1] 加载模型...")
    models, features_map = load_models()
    if not models:
        print("  ✗ 无可用模型，请先运行训练脚本")
        sys.exit(1)

    # ── 确定推理日期 ──────────────────────────────────────────────────────────
    conn = duckdb.connect(DB_PATH, read_only=True)
    print("\n[2] 确定推理日期...")
    infer_date, latest_rebal, next_rebal, warmup_start, is_rebal, all_td = get_inference_date(conn, args.date)
    print(f"  推理日期:     {infer_date}{'  ← 今日为调仓日 ✓' if is_rebal else '  （非调仓日）'}")
    print(f"  上次调仓日:   {latest_rebal}（信号基于此截面）")
    if next_rebal:
        print(f"  下次调仓日:   {next_rebal}")
    print(f"  数据预热起点: {warmup_start}")

    # ── 指数择时 ──────────────────────────────────────────────────────────────
    print("\n[3] 指数择时信号...")
    slots, state_desc = get_timing_slots(infer_date)
    print(f"  大盘状态: {state_desc}")
    if slots is not None:
        max_hold = MAX_SLOTS if slots == 20 else (HALF_SLOTS if slots == 10 else 0)
        print(f"  当前仓位限制: 最多 {max_hold} 只持仓")

    # ── 加载特征数据 ──────────────────────────────────────────────────────────
    print("\n[4] 加载特征数据...")
    tech_df = load_tech_features(conn, warmup_start, infer_date)
    print(f"  技术/估值特征: {len(tech_df):,} 行")

    # 收盘价矩阵（用于 RSI）
    close_pv = tech_df.pivot(index="trade_date", columns="ts_code", values="close")
    rsi_mat = compute_rsi(close_pv)

    mf_df = load_moneyflow(conn, warmup_start, infer_date)
    print(f"  资金流向特征: {len(mf_df):,} 行")

    fund_df = load_fundamental(conn)
    print(f"  基本面PIT面板: {len(fund_df):,} 条")

    stock_info = conn.execute("""
        SELECT ts_code, COALESCE(industry, '未知') AS industry FROM stock_basic
    """).fetchdf()

    # ── 过滤目标日 ────────────────────────────────────────────────────────────
    target_date = latest_rebal   # 用最近调仓日的截面生成信号

    print(f"\n[5] 组装 {target_date} 截面...")
    tech_td = tech_df[tech_df["trade_date"] == target_date].copy()
    if tech_td.empty:
        print(f"  ✗ {target_date} 无技术特征数据")
        conn.close()
        sys.exit(1)

    ts_codes_today = tech_td["ts_code"].tolist()
    print(f"  原始股票数: {len(ts_codes_today)}")

    # RSI
    rsi_today = rsi_mat.loc[target_date] if target_date in rsi_mat.index else pd.Series(dtype=float)
    rsi_df = rsi_today.reset_index().rename(columns={"ts_code": "ts_code", target_date: "rsi_14"})
    tech_td = tech_td.merge(rsi_df, on="ts_code", how="left")

    # 资金流向
    mf_td = mf_df[mf_df["trade_date"] == target_date].copy()
    tech_td = tech_td.merge(mf_td, on=["ts_code", "trade_date"], how="left")
    mv_wan = tech_td["total_mv_100m"] * 10000
    tech_td["mf_1d_mv"]                   = tech_td["mf_1d_raw"]  / mv_wan.replace(0, np.nan)
    tech_td["mf_5d_mv"]                   = tech_td["mf_5d_raw"]  / mv_wan.replace(0, np.nan)
    tech_td["mf_20d_mv"]                  = tech_td["mf_20d_raw"] / mv_wan.replace(0, np.nan)
    tech_td["large_net_5d_ratio"]         = tech_td["large_net_5d"]  / tech_td["total_flow_5d"].replace(0, np.nan)
    tech_td["large_net_20d_ratio"]        = tech_td["large_net_20d"] / tech_td["total_flow_20d"].replace(0, np.nan)
    tech_td["retail_net_5d_ratio"]        = tech_td["retail_net_5d"] / tech_td["total_flow_5d"].replace(0, np.nan)
    tech_td["smart_retail_divergence_5d"] = tech_td["large_net_5d_ratio"] - tech_td["retail_net_5d_ratio"]

    # 基本面 PIT
    fund_pit = pit_join(fund_df, ts_codes_today, target_date)
    fund_cols_keep = ["ts_code"] + [c for c in FUND_COLS if c in fund_pit.columns]
    tech_td = tech_td.merge(fund_pit[fund_cols_keep], on="ts_code", how="left")

    # 分析师
    mv_map = dict(zip(tech_td["ts_code"], tech_td["total_mv_100m"] * 10000))
    analyst_df = load_analyst(conn, target_date, mv_map)
    tech_td = tech_td.merge(analyst_df, on="ts_code", how="left")

    # 股东户数
    holder_pit = load_holder_pit(conn, ts_codes_today, target_date)
    tech_td = tech_td.merge(holder_pit[["ts_code", "holder_chg_qoq"]], on="ts_code", how="left")

    # SUE
    sue_df = load_sue_pit(conn, target_date)
    tech_td = tech_td.merge(sue_df, on="ts_code", how="left")

    # 行业
    tech_td = tech_td.merge(stock_info, on="ts_code", how="left")
    tech_td["industry"] = tech_td["industry"].fillna("未知")

    # 交叉特征
    pe_safe = tech_td["pe_ttm"].clip(lower=5.0)
    tech_td["smart_momentum"]          = tech_td["ret_20d"] * tech_td["large_net_20d_ratio"]
    tech_td["momentum_adj_reversal"]   = tech_td["ret_60d"] - tech_td["ret_5d"]
    tech_td["quality_value_score"]     = tech_td["ni_growth_yoy"] / pe_safe

    print(f"  特征组装完成: {len(tech_td)} 只股票")

    # ── 合规过滤 ──────────────────────────────────────────────────────────────
    eligible = get_eligible_stocks(conn, target_date, all_td)
    tech_td = tech_td[tech_td["ts_code"].isin(eligible)].copy()
    print(f"  合规过滤后: {len(tech_td)} 只股票")

    # 非停牌
    vol_today = conn.execute(f"""
        SELECT ts_code FROM daily_price
        WHERE trade_date = '{infer_date}' AND vol > 0
    """).fetchdf()["ts_code"].tolist()
    tech_td = tech_td[tech_td["ts_code"].isin(vol_today)].copy()
    print(f"  去除停牌后: {len(tech_td)} 只股票")

    # ── 涨停过滤 ──────────────────────────────────────────────────────────────
    limit_up = conn.execute(f"""
        SELECT ts_code FROM daily_price
        WHERE trade_date = '{infer_date}' AND pct_chg >= 9.5
    """).fetchdf()["ts_code"].tolist()
    tech_td = tech_td[~tech_td["ts_code"].isin(limit_up)].copy()
    if limit_up:
        print(f"  去除涨停: {len(limit_up)} 只 → 剩余 {len(tech_td)} 只")

    conn.close()

    # ── 预处理 ────────────────────────────────────────────────────────────────
    print("\n[6] 预处理（MAD → 中性化 → Z-score）...")
    avail_feats = [c for c in ALL_FEATURES if c in tech_td.columns]
    feature_na_rate = tech_td[avail_feats].isna().mean(axis=1)
    tech_td = tech_td[feature_na_rate < 0.50].copy()

    for col in avail_feats:
        tech_td[col] = tech_td.groupby("industry")[col].transform(
            lambda x: x.fillna(x.median())
        ).fillna(0)

    tech_td = preprocess_single_date(tech_td, avail_feats)
    print(f"  预处理完成: {len(tech_td)} 只股票，{len(avail_feats)} 个特征")

    # ── 模型推理 ──────────────────────────────────────────────────────────────
    print("\n[7] 模型推理...")
    scored = predict_scores(models, features_map, tech_td)
    scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
    print(f"  推理完成: {len(scored)} 只股票已排序")

    # ── 持仓退出信号 ──────────────────────────────────────────────────────────
    holdings = {}
    if args.holdings:
        p = Path(args.holdings)
        if p.exists():
            with open(p) as f:
                holdings = json.load(f)
            print(f"\n[8] 当前持仓退出检查（{len(holdings)} 只）...")
        else:
            print(f"\n[8] 持仓文件不存在: {args.holdings}")

    exits = []
    if holdings:
        conn2 = duckdb.connect(DB_PATH, read_only=True)
        exits = check_exits(holdings, conn2, infer_date, all_td)
        conn2.close()

    # ── 输出结果 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    next_desc = f"，下次调仓: {next_rebal}" if next_rebal else ""
    print(f"  推理日期: {infer_date}  ({'调仓日 ✓' if is_rebal else '非调仓日' + next_desc})")
    print(f"  大盘信号: {state_desc}")
    print("=" * 65)

    # 退出信号
    if holdings:
        print(f"\n{'─'*65}")
        print(f"  ⚠  退出信号 ({len(exits)} 只)")
        print(f"{'─'*65}")
        if exits:
            for e in exits:
                print(f"  卖出 {e['ts_code']:12s}  {e['reason']:20s}  "
                      f"当前 ¥{e['current_price']:.2f}  "
                      f"成本 ¥{e['entry_price']:.2f}  "
                      f"{'↑' if e['pnl_pct']>=0 else '↓'}{abs(e['pnl_pct']):.1f}%  "
                      f"持有{e['hold_days']}天")
        else:
            print("  ✓ 所有持仓均未触发退出条件")

    # 当前持仓状态
    if holdings:
        print(f"\n{'─'*65}")
        print(f"  当前持仓（共 {len(holdings)} 只）")
        print(f"{'─'*65}")
        for ts, info in holdings.items():
            ep = float(info.get("entry_price", 0))
            ed = info.get("entry_date", "?")
            print(f"  持有 {ts:12s}  入场 ¥{ep:.2f}  日期 {ed}")

    # 买入候选
    if slots == 0:
        print(f"\n  ⛔ 大盘信号为熊市/空仓，不建议开新仓")
    else:
        n_current = len(holdings)
        actual_max = MAX_SLOTS if slots == 20 else HALF_SLOTS
        n_available_slots = max(0, actual_max - n_current)

        print(f"\n{'─'*65}")
        print(f"  🚀  买入候选（当前持仓 {n_current}，最大 {actual_max}，可开 {n_available_slots} 个槽位）")
        print(f"{'─'*65}")
        print(f"  {'排名':>4}  {'代码':12s}  {'得分':>6}  {'H10排名':>7}  {'H5排名':>7}  {'市值(亿)':>8}")
        print(f"  {'-'*55}")

        shown = 0
        for i, row in scored.iterrows():
            if shown >= args.top_k:
                break
            if row["ts_code"] in holdings:
                continue  # 已持有
            mktcap = row.get("total_mv_100m", float("nan"))
            print(f"  {shown+1:>4}  {row['ts_code']:12s}  {row['score']:.4f}  "
                  f"{row.get('r10', float('nan')):>7.4f}  "
                  f"{row.get('r5',  float('nan')):>7.4f}  "
                  f"{mktcap:>8.1f}")
            shown += 1

    print(f"\n{'═'*65}")
    print(f"  说明：得分为 H10×80% + H5×20% 截面百分位，越高越优")
    # 从已保存的 features_h10.json 中读取 train_end 显示
    try:
        with open(MODELS_DIR / "features_h10.json") as _f:
            _meta = json.load(_f)
        _train_end = _meta.get("train_end", "?")
    except Exception:
        _train_end = "?"
    print(f"  该信号基于训练截止 {_train_end} 的模型")
    print(f"  T日收盘执行，实际操作请在明日开盘前下单")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
