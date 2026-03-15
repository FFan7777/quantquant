#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略级超参数搜索  Strategy Hyperparameter Search
=======================================================

严格的三段式数据划分（与截面选股模型一致）:
  Train end: 2023-12-31  （模型训练，不参与策略 HP 搜索）
  Val:       2024-02-01 ~ 2024-12-31  ← 策略 HP 搜索在此期间运行
  Test:      2025-02-01 ~ 2026-03-15  ← 唯一真实 OOS 评测

搜索参数（坐标下降，共 ~30 次回测）:
  Step A: (max_slots, half_slots) — 持仓容量
  Step B: stop_loss               — 止损幅度
  Step C: (ma_death_days, min_hold_days) — MA死叉退出参数
  Step D: mktcap_pct_cut          — 市值分位过滤

数据依赖:
  output/csv/xgb_cs_pred_val.csv    ← xgboost_cross_section.py 生成（train-only模型）
  output/csv/xgb_cross_section_predictions.csv  ← 同上（train+val模型）
  output/csv/index_timing_predictions.csv       ← index_timing_model.py 生成
"""

import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from data_collect.config import config

# ══════════════════════════════════════════════════════════════════════
# 1. 全局路径与回测期定义
# ══════════════════════════════════════════════════════════════════════
DB_PATH = config.db_path

VAL_START  = "20240201"   # 含20日隔离（train end 2023-12-31 + embargo）
VAL_END    = "20241231"
TEST_START = "20250201"   # 含20日隔离（val end 2024-12-31 + embargo）
TEST_END   = "20260315"

PRICE_WARMUP = "20221101"   # MA 计算预热期

INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE  = 0.001
TRANSFER_RATE   = 0.00002
SLIPPAGE_RATE   = 0.0001
MIN_COMMISSION  = 5.0
MIN_TRADE_VALUE = 2000.0
RISK_FREE_RATE  = 0.02

OUT_DIR = ROOT / "output" / "strategy_hp_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 2. 策略参数 dataclass（替代全局变量）
# ══════════════════════════════════════════════════════════════════════
@dataclass
class StrategyParams:
    max_slots:       int   = 8      # 满仓最多持股数
    half_slots:      int   = 3      # 半仓最多持股数（slots=10 时）
    stop_loss:       float = 0.08   # 距入场价回撤 X% → 硬止损
    trailing_stop:   float = 1.00   # 追踪止损（1.0=禁用）
    min_hold_days:   int   = 5      # 最短持有期（交易日），之后才检查MA死叉
    ma_death_days:   int   = 5      # MA5<MA20 连续 N 天 → 死叉退出
    mktcap_pct_cut:  int   = 10     # 排除市值后 X% 的微小盘股
    min_listed_days: int   = 90     # 上市不满 N 天排除
    slot_confirm_days: int = 3      # 连续 N 天 slots>0 才开新仓（防 whipsaw）
    use_vol_scale:   bool  = True   # 风险平价加权
    use_signal_scale: bool = True   # 信号强度加权
    h10_weight:      float = 1.0    # H10 预测权重（0~1），余下给 H5

    def label(self) -> str:
        return (f"ms{self.max_slots}_hs{self.half_slots}_"
                f"sl{int(self.stop_loss*100)}_"
                f"md{self.ma_death_days}_mh{self.min_hold_days}_"
                f"mc{self.mktcap_pct_cut}")


DEFAULT_PARAMS = StrategyParams()


# ══════════════════════════════════════════════════════════════════════
# 3. 数据加载（一次性，供所有回测复用）
# ══════════════════════════════════════════════════════════════════════
def _norm_date(s) -> str:
    return str(s).replace('-', '')


def load_price_data(end_date: str) -> Tuple:
    """加载价格矩阵（含预热），返回 (price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv)"""
    print("  加载价格数据...")
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        df = conn.execute(f"""
            SELECT trade_date, ts_code, close, vol
            FROM daily_price
            WHERE trade_date >= '{PRICE_WARMUP}'
              AND trade_date <= '{end_date}'
              AND ts_code NOT LIKE '8%'
              AND ts_code NOT LIKE '4%'
            ORDER BY trade_date, ts_code
        """).fetchdf()

    df['trade_date'] = df['trade_date'].apply(_norm_date)
    price_pv = df.pivot(index='trade_date', columns='ts_code', values='close')
    vol_pv   = df.pivot(index='trade_date', columns='ts_code', values='vol')

    ma5_pv      = price_pv.rolling(5,  min_periods=3).mean()
    ma20_pv     = price_pv.rolling(20, min_periods=10).mean()
    retvol20_pv = price_pv.pct_change().rolling(20, min_periods=10).std()

    pct_chg_pv = price_pv.pct_change()
    print(f"    价格矩阵: {len(price_pv)} 天 × {len(price_pv.columns):,} 只")
    return price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv


def load_safety_info(params: StrategyParams, rebal_dates: List[str]) -> Dict[str, set]:
    """预加载所有调仓日的合规股票集合（ST/市值/上市天数过滤）"""
    print("  加载合规股信息...")
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        sb = conn.execute("SELECT ts_code, list_date, name FROM stock_basic").fetchdf()
        dates_sql = "'" + "','".join(rebal_dates) + "'"
        mktcap_df = conn.execute(f"""
            SELECT ts_code, trade_date, total_mv
            FROM daily_basic
            WHERE trade_date IN ({dates_sql}) AND total_mv > 0
        """).fetchdf()

    sb['is_st'] = sb['name'].str.contains('ST', na=False)
    sb['list_date'] = sb['list_date'].apply(_norm_date)
    st_set = set(sb.loc[sb['is_st'], 'ts_code'])
    list_date_map = dict(zip(sb['ts_code'], sb['list_date']))
    mktcap_df['trade_date'] = mktcap_df['trade_date'].apply(_norm_date)

    eligible: Dict[str, set] = {}
    for date in rebal_dates:
        sub = mktcap_df[mktcap_df['trade_date'] == date]
        if sub.empty:
            eligible[date] = set()
            continue
        cutoff = np.percentile(sub['total_mv'].values, params.mktcap_pct_cut)
        elig = set()
        for _, row in sub.iterrows():
            ts = row['ts_code']
            if ts in st_set or row['total_mv'] <= cutoff:
                continue
            ld = list_date_map.get(ts, '20000101')
            try:
                if (pd.Timestamp(date) - pd.Timestamp(ld)).days < params.min_listed_days:
                    continue
            except Exception:
                continue
            elig.add(ts)
        eligible[date] = elig
    return eligible


def load_cs_predictions(period: str = "val") -> pd.DataFrame:
    """加载截面选股预测。period='val' 或 'test'"""
    if period == "val":
        fpath = ROOT / "output" / "csv" / "xgb_cs_pred_val.csv"
        start, end = VAL_START, VAL_END
    else:
        fpath = ROOT / "output" / "csv" / "xgb_cross_section_predictions.csv"
        start, end = TEST_START, TEST_END

    if not fpath.exists():
        raise FileNotFoundError(
            f"找不到预测文件: {fpath}\n"
            "请先运行 xgboost_cross_section.py"
        )
    df = pd.read_csv(fpath, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    df = df[(df['trade_date'] >= start) & (df['trade_date'] <= end)]
    print(f"  CS预测({period}): {len(df):,}行  {df['trade_date'].nunique()}个截面  "
          f"{df['ts_code'].nunique()}只股票")
    return df[['ts_code', 'trade_date', 'pred']].copy()


def load_index_timing(period: str = "val") -> pd.Series:
    """加载指数择时信号 slots∈{0,10,20}"""
    fpath = ROOT / "output" / "csv" / "index_timing_predictions.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"找不到: {fpath}\n请先运行 index_timing_model.py")
    df = pd.read_csv(fpath, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    start = VAL_START if period == "val" else TEST_START
    end   = VAL_END   if period == "val" else TEST_END
    df = df[(df['trade_date'] >= start) & (df['trade_date'] <= end)]
    slots = df.set_index('trade_date')['slots'].astype(int)
    print(f"  指数择时({period}): 0:{(slots==0).sum()} / 10:{(slots==10).sum()} / 20:{(slots==20).sum()}")
    return slots


def load_csi300(period: str = "val") -> pd.Series:
    start = VAL_START if period == "val" else TEST_START
    end   = VAL_END   if period == "val" else TEST_END
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        df = conn.execute(f"""
            SELECT trade_date, close FROM index_daily
            WHERE ts_code='000300.SH'
              AND trade_date >= '{start}' AND trade_date <= '{end}'
            ORDER BY trade_date
        """).fetchdf()
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    return df.set_index('trade_date')['close']


# ══════════════════════════════════════════════════════════════════════
# 4. 参数化回测引擎（替代 index_ma_combined_strategy 中的全局变量）
# ══════════════════════════════════════════════════════════════════════
def trade_cost(value: float, is_sell: bool) -> float:
    v = abs(value)
    c = max(v * COMMISSION_RATE, MIN_COMMISSION)
    t = v * STAMP_TAX_RATE if is_sell else 0.0
    return c + t + v * TRANSFER_RATE + v * SLIPPAGE_RATE


class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.shares:      Dict[str, int]   = {}
        self.entry_price: Dict[str, float] = {}
        self.entry_date:  Dict[str, str]   = {}
        self.peak_price:  Dict[str, float] = {}
        self.hold_days:   Dict[str, int]   = {}
        self.dc_count:    Dict[str, int]   = {}

    def total_value(self, prices: pd.Series) -> float:
        pos_val = sum(self.shares.get(ts, 0) * float(prices.get(ts, 0) or 0)
                      for ts in list(self.shares))
        return self.cash + pos_val

    def sell(self, ts_code: str, price: float, date: str) -> float:
        sh = self.shares.pop(ts_code, 0)
        if sh == 0 or not np.isfinite(price) or price <= 0:
            for d in [self.entry_price, self.entry_date, self.peak_price,
                      self.hold_days, self.dc_count]:
                d.pop(ts_code, None)
            return 0.0
        val = sh * price
        cash_in = val - trade_cost(val, is_sell=True)
        self.cash += cash_in
        for d in [self.entry_price, self.entry_date, self.peak_price,
                  self.hold_days, self.dc_count]:
            d.pop(ts_code, None)
        return cash_in

    def buy(self, ts_code: str, price: float, target_val: float, date: str) -> bool:
        if not np.isfinite(price) or price <= 0 or target_val < MIN_TRADE_VALUE:
            return False
        cost_factor = 1 + COMMISSION_RATE + TRANSFER_RATE + SLIPPAGE_RATE
        affordable = min(target_val, self.cash / cost_factor)
        if affordable < MIN_TRADE_VALUE:
            return False
        sh = int(affordable / price / 100) * 100
        if sh == 0:
            sh = max(1, int(affordable / price))
        val = sh * price
        total = val + trade_cost(val, is_sell=False)
        if total > self.cash:
            sh = int((self.cash / cost_factor) / price)
            if sh == 0:
                return False
            val = sh * price
            total = val + trade_cost(val, is_sell=False)
        self.cash -= total
        old_sh = self.shares.get(ts_code, 0)
        old_ep = self.entry_price.get(ts_code, price)
        new_sh = old_sh + sh
        self.shares[ts_code]      = new_sh
        self.entry_price[ts_code] = (old_ep * old_sh + price * sh) / new_sh
        self.entry_date[ts_code]  = self.entry_date.get(ts_code, date)
        self.peak_price[ts_code]  = max(self.peak_price.get(ts_code, price), price)
        self.hold_days[ts_code]   = self.hold_days.get(ts_code, 0)
        self.dc_count[ts_code]    = self.dc_count.get(ts_code, 0)
        return True

    def update_hold_days(self):
        for ts in list(self.hold_days):
            self.hold_days[ts] = self.hold_days.get(ts, 0) + 1


def daily_stop_check(portfolio: Portfolio, prices: pd.Series,
                     ma5: pd.Series, ma20: pd.Series,
                     p: StrategyParams) -> List[str]:
    """检查止损/止盈/MA死叉，返回需卖出的 ts_code 列表"""
    to_sell = []
    for ts in list(portfolio.shares):
        price = prices.get(ts, np.nan)
        if pd.isna(price) or price <= 0:
            to_sell.append(ts)
            continue
        ep   = portfolio.entry_price.get(ts, price)
        peak = portfolio.peak_price.get(ts, price)
        if price > peak:
            portfolio.peak_price[ts] = price
            peak = price
        # 硬止损
        if price < ep * (1 - p.stop_loss):
            to_sell.append(ts)
            continue
        # 追踪止损（默认禁用）
        if price < peak * (1 - p.trailing_stop):
            to_sell.append(ts)
            continue
        # MA 死叉（需满足 min_hold_days）
        hd = portfolio.hold_days.get(ts, 0)
        if hd >= p.min_hold_days:
            m5  = ma5.get(ts, np.nan)
            m20 = ma20.get(ts, np.nan)
            if pd.notna(m5) and pd.notna(m20):
                if m5 < m20:
                    portfolio.dc_count[ts] = portfolio.dc_count.get(ts, 0) + 1
                    if portfolio.dc_count[ts] >= p.ma_death_days:
                        to_sell.append(ts)
                        continue
                else:
                    portfolio.dc_count[ts] = 0
    return list(set(to_sell))


def rebalance(portfolio: Portfolio, target_stocks: List[str],
              n_slots: int, prices: pd.Series, date: str,
              slot_confirmed: bool, ret_vol: pd.Series,
              p: StrategyParams) -> int:
    """按 CS 打分调仓。n_slots=0 不开仓。"""
    if n_slots == 0:
        return 0
    total_val  = portfolio.total_value(prices)
    target_set = set(target_stocks[:n_slots])
    trades = 0
    # 卖出不在目标中的持仓（已过 min_hold_days）
    for ts in sorted(set(portfolio.shares) - target_set):
        if portfolio.hold_days.get(ts, 0) >= p.min_hold_days:
            price = prices.get(ts, np.nan)
            if pd.notna(price) and price > 0:
                portfolio.sell(ts, price, date)
                trades += 1
    # 买入
    if slot_confirmed:
        target_list = target_stocks[:n_slots]
        top_stocks  = [ts for ts in target_list if ts not in portfolio.shares]
        fixed_sv    = total_val / p.max_slots

        # 信号强度加权
        n = len(target_list)
        if p.use_signal_scale and n > 0:
            sig_map = {ts: (1.5 - 1.0 * i / max(n - 1, 1)) for i, ts in enumerate(target_list)}
        else:
            sig_map = {ts: 1.0 for ts in target_list}

        # 风险平价加权
        if p.use_vol_scale and ret_vol is not None and top_stocks:
            vols = np.array([float(ret_vol.get(ts, np.nan) or np.nan) for ts in top_stocks])
            med  = float(np.nanmedian(vols)) if not np.all(np.isnan(vols)) else 0.02
            vols = np.where(np.isnan(vols), med, vols)
            vols = np.maximum(vols, 0.005)
            vol_scale = np.clip(med / vols, 0.5, 2.0)
            slot_vals = {ts: fixed_sv * sig_map.get(ts, 1.0) * vol_scale[i]
                         for i, ts in enumerate(top_stocks)}
        else:
            slot_vals = {ts: fixed_sv * sig_map.get(ts, 1.0) for ts in top_stocks}

        for ts in top_stocks:
            p_ = prices.get(ts, np.nan)
            if pd.isna(p_) or p_ <= 0:
                continue
            if portfolio.buy(ts, p_, slot_vals[ts], date):
                trades += 1
    return trades


def _get_slots(slots_series: pd.Series, date: str, default: int) -> int:
    if date in slots_series.index:
        return int(slots_series[date])
    prior = slots_series[slots_series.index <= date]
    return int(prior.iloc[-1]) if len(prior) > 0 else default


def run_backtest(
    p:            StrategyParams,
    cs_preds:     pd.DataFrame,
    index_slots:  pd.Series,
    price_pv:     pd.DataFrame,
    ma5_pv:       pd.DataFrame,
    ma20_pv:      pd.DataFrame,
    vol_pv:       pd.DataFrame,
    retvol20_pv:  pd.DataFrame,
    pct_chg_pv:   pd.DataFrame,
    eligible:     Dict[str, set],
    start_date:   str,
    end_date:     str,
    verbose:      bool = False,
) -> pd.DataFrame:
    """
    日频回测主循环（完全参数化，不依赖全局变量）。
    返回 equity_df（含 total_value, cash, n_positions, slots 列）。
    """
    pf = Portfolio(INITIAL_CAPITAL)
    cs_by_date = {dt: grp for dt, grp in cs_preds.groupby('trade_date')}
    rebal_set  = set(cs_preds['trade_date'].unique())
    all_dates  = sorted(d for d in price_pv.index if start_date <= d <= end_date)

    equity = []
    consec_nonzero = 0

    for date in all_dates:
        slots_today  = _get_slots(index_slots, date, p.half_slots)
        bear_mode    = (slots_today == 0)
        if slots_today > 0:
            consec_nonzero += 1
        else:
            consec_nonzero = 0

        prices = price_pv.loc[date]
        ma5    = ma5_pv.loc[date]  if date in ma5_pv.index  else pd.Series(dtype=float)
        ma20   = ma20_pv.loc[date] if date in ma20_pv.index else pd.Series(dtype=float)

        # 每日风控
        exits = daily_stop_check(pf, prices, ma5, ma20, p)
        for ts in exits:
            pr = prices.get(ts, np.nan)
            pr = float(pr if pd.notna(pr) and pr > 0 else pf.entry_price.get(ts, 1.0))
            pf.sell(ts, pr, date)

        # 调仓日
        if date in rebal_set and slots_today > 0:
            elig_set  = eligible.get(date, set())
            vol_today = vol_pv.loc[date] if date in vol_pv.index else pd.Series(dtype=float)
            active    = set(vol_today[vol_today > 0].dropna().index)

            scores_df = cs_by_date.get(date, pd.DataFrame(columns=['ts_code', 'pred']))
            scores_df = scores_df[
                scores_df['ts_code'].isin(elig_set) &
                scores_df['ts_code'].isin(active)
            ].sort_values('pred', ascending=False)

            # 涨停过滤
            if date in pct_chg_pv.index:
                limit_up = set(pct_chg_pv.loc[date][pct_chg_pv.loc[date] >= 0.095].dropna().index)
                scores_df = scores_df[~scores_df['ts_code'].isin(limit_up)]

            target_stocks = scores_df['ts_code'].tolist()
            slot_confirmed = (consec_nonzero >= p.slot_confirm_days)
            actual_slots   = p.max_slots if slots_today == 20 else p.half_slots

            if target_stocks:
                rv = retvol20_pv.loc[date] if date in retvol20_pv.index else None
                rebalance(pf, target_stocks, actual_slots, prices, date,
                          slot_confirmed, rv, p)

        pf.update_hold_days()
        pf.cash *= (1 + RISK_FREE_RATE / 252)

        tv = pf.total_value(prices)
        equity.append({'date': date, 'total_value': tv,
                        'cash': pf.cash, 'n_positions': len(pf.shares),
                        'slots': slots_today})

    equity_df = pd.DataFrame(equity).set_index('date')
    equity_df.index = pd.to_datetime(equity_df.index, format='%Y%m%d')
    return equity_df


# ══════════════════════════════════════════════════════════════════════
# 5. 绩效指标
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(equity_df: pd.DataFrame) -> dict:
    vals = equity_df['total_value']
    rets = vals.pct_change().dropna()
    n = len(rets)
    tpy = 244
    yrs = n / tpy
    if yrs <= 0 or len(vals) < 2:
        return dict(annual_ret=0, max_dd=0, sharpe=0, calmar=0, total_ret=0, years=0,
                    annual_vol=0, annual_rets={})

    total_ret  = vals.iloc[-1] / vals.iloc[0] - 1
    annual_ret = (1 + total_ret) ** (1 / yrs) - 1
    annual_vol = rets.std() * np.sqrt(tpy)
    rf_daily   = 0.02 / tpy
    sharpe     = ((rets.mean() - rf_daily) / rets.std() * np.sqrt(tpy)
                  if rets.std() > 0 else 0)
    peak  = vals.expanding().max()
    dd    = (vals - peak) / peak
    max_dd = float(dd.min())
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.nan

    annual_rets = {}
    for yr in range(2023, 2027):
        mask = equity_df.index.year == yr
        if mask.sum() < 5:
            continue
        yv = vals[mask]
        annual_rets[str(yr)] = yv.iloc[-1] / yv.iloc[0] - 1

    return dict(total_ret=total_ret, annual_ret=annual_ret, annual_vol=annual_vol,
                sharpe=sharpe, calmar=calmar, max_dd=max_dd, years=yrs,
                annual_rets=annual_rets)


def score(m: dict) -> float:
    """主优化目标：卡玛比率（年化收益/最大回撤），同时要求正收益"""
    if m['annual_ret'] <= 0:
        return -999.0
    return m['calmar'] if np.isfinite(m['calmar']) else 0.0


# ══════════════════════════════════════════════════════════════════════
# 6. 超参数搜索：坐标下降
# ══════════════════════════════════════════════════════════════════════
def run_one(label: str, p: StrategyParams,
            cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
            vol_pv, retvol20_pv, pct_chg_pv, eligible,
            start_date: str, end_date: str) -> dict:
    eq = run_backtest(p, cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
                      vol_pv, retvol20_pv, pct_chg_pv, eligible,
                      start_date, end_date)
    m = compute_metrics(eq)
    m['label'] = label
    m['params'] = asdict(p)
    m['equity_df'] = eq
    return m


def search_dimension(name: str, dim_overrides: list,
                     base: StrategyParams, cs_preds, index_slots,
                     price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
                     pct_chg_pv, eligible,
                     start_date: str, end_date: str,
                     total_counter: list) -> StrategyParams:
    """
    在 dim_overrides（dict列表）上枚举，每个 override 只覆盖搜索维度的字段。
    其余字段继承 base（保留前序步骤累积的最优值）。
    """
    print(f"\n── 搜索维度: {name}  base={base.label()} ──")
    results = []
    for override in dim_overrides:
        p = StrategyParams(**{**asdict(base), **override})
        m = run_one(p.label(), p, cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
                    vol_pv, retvol20_pv, pct_chg_pv, eligible, start_date, end_date)
        results.append(m)
        total_counter[0] += 1
        print(f"  [{total_counter[0]:2d}] {p.label():<52s}  "
              f"ann={m['annual_ret']:>+.1%}  DD={m['max_dd']:>.1%}  "
              f"Sharpe={m['sharpe']:>.2f}  Calmar={m['calmar']:>.2f}  "
              f"score={score(m):>.2f}")

    best = max(results, key=lambda x: score(x))
    print(f"  → 最优: {best['label']}  Calmar={best['calmar']:.2f}")
    best_p = StrategyParams(**best['params'])
    return best_p, results


def sensitivity_test(best_p: StrategyParams, cs_preds, index_slots,
                     price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
                     pct_chg_pv, eligible, start_date: str, end_date: str) -> pd.DataFrame:
    """
    对每个核心参数做±1步扰动，验证稳健性。
    Calmar 变化 > 0.5 视为不稳定。
    """
    print("\n[敏感性测试] 对最优参数逐维度扰动（Calmar变化>0.5视为不稳定）")
    INSTABILITY = 0.5

    base_m = run_one("base", best_p, cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
                     vol_pv, retvol20_pv, pct_chg_pv, eligible, start_date, end_date)
    base_calmar = base_m['calmar']
    print(f"  基准 Calmar={base_calmar:.3f}  params={best_p}")

    dims = {
        "max_slots":     [4, 6, 8, 10, 12, 15],
        "half_slots":    [2, 3, 4, 5, 6],
        "stop_loss":     [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15],
        "ma_death_days": [2, 3, 4, 5, 6, 7, 10],
        "min_hold_days": [2, 3, 4, 5, 6, 7, 10],
    }

    rows = []
    for dim, candidates in dims.items():
        if not hasattr(best_p, dim):
            continue
        best_val = getattr(best_p, dim)
        for v in candidates:
            test_params = StrategyParams(**{**asdict(best_p), dim: v})
            # half_slots must be < max_slots
            if test_params.half_slots >= test_params.max_slots:
                continue
            m = run_one(f"{dim}={v}", test_params, cs_preds, index_slots,
                        price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
                        pct_chg_pv, eligible, start_date, end_date)
            delta  = m['calmar'] - base_calmar
            stable = abs(delta) <= INSTABILITY
            flag   = "✓" if stable else "⚠ 不稳定"
            is_best = (v == best_val)
            print(f"  {dim}={v:<6}{'[best]' if is_best else '      '} "
                  f"Calmar={m['calmar']:>+.3f}  Δ={delta:>+.3f}  {flag}")
            rows.append({'dim': dim, 'value': v, 'is_best': is_best,
                         'calmar': m['calmar'], 'delta': delta,
                         'annual_ret': m['annual_ret'], 'max_dd': m['max_dd'],
                         'stable': stable})

        # 汇总该维度
        dim_rows = [r for r in rows if r['dim'] == dim]
        calmars  = [r['calmar'] for r in dim_rows if np.isfinite(r['calmar'])]
        rng = max(calmars) - min(calmars) if calmars else 0
        stab = "✓ 稳健" if rng <= INSTABILITY else "⚠ 敏感"
        print(f"  → {dim}: Calmar范围={rng:.3f}  {stab}\n")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 7. 可视化
# ══════════════════════════════════════════════════════════════════════
def plot_equity(val_eq: pd.DataFrame, test_eq: pd.DataFrame,
                csi300_val: pd.Series, csi300_test: pd.Series,
                best_p: StrategyParams):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Strategy HP Search: best params = {best_p.label()}", fontsize=11)

    for ax, eq, csi, period in [
        (axes[0], val_eq, csi300_val, "Val 2024"),
        (axes[1], test_eq, csi300_test, "Test 2025+"),
    ]:
        nav = eq['total_value'] / eq['total_value'].iloc[0]
        csi_norm = csi / csi.iloc[0] if len(csi) > 0 else pd.Series()
        ax.plot(nav.index, nav.values, label="Strategy", color="steelblue", linewidth=1.5)
        if len(csi_norm) > 0:
            ax.plot(pd.to_datetime(csi_norm.index, format='%Y%m%d'),
                    csi_norm.values, label="CSI300", color="gray", linewidth=1, alpha=0.7)
        ax.axhline(1, color='black', linewidth=0.6)
        m = compute_metrics(eq)
        ax.set_title(f"{period}  ann={m['annual_ret']:+.1%}  DD={m['max_dd']:.1%}  "
                     f"Calmar={m['calmar']:.2f}", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylabel("归一化净值")

    plt.tight_layout()
    out = OUT_DIR / "strategy_hp_equity.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  净值图已保存: {out}")


def plot_sensitivity(sens_df: pd.DataFrame, period_label: str):
    dims = sens_df['dim'].unique()
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 4))
    if len(dims) == 1:
        axes = [axes]
    fig.suptitle(f"敏感性测试 ({period_label})", fontsize=11)

    for ax, dim in zip(axes, dims):
        sub = sens_df[sens_df['dim'] == dim].sort_values('value')
        colors = ['#d62728' if not r['stable'] else '#1f77b4' for _, r in sub.iterrows()]
        ax.bar(sub['value'].astype(str), sub['calmar'], color=colors)
        best_v = sub.loc[sub['is_best'], 'value'].values
        if len(best_v):
            ax.axvline(str(best_v[0]), color='orange', linewidth=2, linestyle='--',
                       label=f'best={best_v[0]}')
        ax.set_title(dim, fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("Calmar")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = OUT_DIR / "strategy_hp_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  敏感性图已保存: {out}")


# ══════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════
def main():
    import time
    t0 = time.time()
    print("=" * 65)
    print("  策略超参数搜索（坐标下降 on Val=2024，严格 OOS = Test 2025+）")
    print("=" * 65)

    # ── 加载数据（一次性）──────────────────────────────────────────────
    print("\n[数据加载]")
    price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv = \
        load_price_data(TEST_END)
    cs_val  = load_cs_predictions("val")
    cs_test = load_cs_predictions("test")
    slot_val  = load_index_timing("val")
    slot_test = load_index_timing("test")
    csi300_val  = load_csi300("val")
    csi300_test = load_csi300("test")

    # 提取所有调仓日（val + test 合并，for 安全过滤加载）
    all_rebal_dates = sorted(set(cs_val['trade_date'].unique()) |
                             set(cs_test['trade_date'].unique()))
    print(f"  调仓日: val={cs_val['trade_date'].nunique()}  "
          f"test={cs_test['trade_date'].nunique()}")

    # 加载安全过滤（使用默认市值参数；mktcap_pct_cut 搜索时会动态重算）
    eligible_val  = load_safety_info(DEFAULT_PARAMS, all_rebal_dates)
    eligible_test = eligible_val  # 同一 DB，共用

    counter = [0]   # mutable counter for numbering

    # ── Step A: (max_slots, half_slots) ───────────────────────────────
    slot_overrides = [
        {'max_slots': ms, 'half_slots': hs}
        for ms, hs in [(4,2), (6,2), (6,3), (8,2), (8,3), (8,4), (10,3), (10,4), (10,5)]
    ]
    best_p, res_A = search_dimension(
        "max_slots × half_slots", slot_overrides, DEFAULT_PARAMS,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv,
        eligible_val, VAL_START, VAL_END, counter)

    # ── Step B: stop_loss ─────────────────────────────────────────────
    stop_overrides = [{'stop_loss': s} for s in [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]]
    best_p, res_B = search_dimension(
        "stop_loss", stop_overrides, best_p,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv,
        eligible_val, VAL_START, VAL_END, counter)

    # ── Step C: (ma_death_days, min_hold_days) ────────────────────────
    ma_overrides = [
        {'ma_death_days': md, 'min_hold_days': mh}
        for md, mh in [(2,5), (3,3), (3,5), (3,7), (5,3), (5,5), (5,7), (5,10), (7,5), (7,7)]
    ]
    best_p, res_C = search_dimension(
        "ma_death_days × min_hold_days", ma_overrides, best_p,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv,
        eligible_val, VAL_START, VAL_END, counter)

    # ── Step D: mktcap_pct_cut ────────────────────────────────────────
    mc_overrides = [{'mktcap_pct_cut': mc} for mc in [5, 10, 15, 20]]
    best_p, res_D = search_dimension(
        "mktcap_pct_cut", mc_overrides, best_p,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv,
        eligible_val, VAL_START, VAL_END, counter)

    # ── 汇总所有搜索结果 ──────────────────────────────────────────────
    all_results = res_A + res_B + res_C + res_D
    results_df = pd.DataFrame([
        {k: v for k, v in m.items() if k not in ('params', 'equity_df', 'annual_rets')}
        for m in all_results
    ])
    results_df.to_csv(OUT_DIR / "hp_search_results.csv", index=False)

    print(f"\n\n{'='*65}")
    print(f"  最优策略超参（Val 2024）: {best_p}")
    print(f"{'='*65}")

    # ── 敏感性测试（在 Val 上）───────────────────────────────────────
    print(f"\n[敏感性测试] 在 Val (2024) 上逐维度扰动")
    sens_df = sensitivity_test(
        best_p, cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv,
        retvol20_pv, pct_chg_pv, eligible_val, VAL_START, VAL_END)
    sens_df.to_csv(OUT_DIR / "sensitivity_results.csv", index=False)

    # ── Val 最终评估 ──────────────────────────────────────────────────
    print("\n[Val 最终评估]")
    val_m = run_one("val_final", best_p, cs_val, slot_val, price_pv, ma5_pv, ma20_pv,
                    vol_pv, retvol20_pv, pct_chg_pv, eligible_val, VAL_START, VAL_END)
    val_eq = val_m['equity_df']

    # ── Test 评估（唯一真实 OOS）──────────────────────────────────────
    print("\n[Test OOS 评估] 使用 Val 搜出的最优参数，不再修改")
    test_m = run_one("test_final", best_p, cs_test, slot_test, price_pv, ma5_pv, ma20_pv,
                     vol_pv, retvol20_pv, pct_chg_pv, eligible_test, TEST_START, TEST_END)
    test_eq = test_m['equity_df']

    # ── 对比基准（默认参数在 Test 上）────────────────────────────────
    print("\n[参考] 默认参数在 Test 上的表现（手动调优的起点）:")
    default_m = run_one("default_params", DEFAULT_PARAMS, cs_test, slot_test,
                        price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv,
                        eligible_test, TEST_START, TEST_END)

    # ── 打印汇总报告 ──────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  策略超参搜索报告")
    print("═" * 65)
    print(f"\n  最优参数:")
    print(f"    max_slots={best_p.max_slots}  half_slots={best_p.half_slots}")
    print(f"    stop_loss={best_p.stop_loss:.0%}  ma_death_days={best_p.ma_death_days}")
    print(f"    min_hold_days={best_p.min_hold_days}  mktcap_pct_cut={best_p.mktcap_pct_cut}")

    for label, m in [("Val (2024) [HP选择集]", val_m),
                     ("Test (2025+) [真实OOS]", test_m),
                     ("Test (默认参数) [参考]", default_m)]:
        print(f"\n  ── {label} ──")
        print(f"    总收益:  {m['total_ret']:+.2%}")
        print(f"    年化:    {m['annual_ret']:+.2%}")
        print(f"    最大DD:  {m['max_dd']:.2%}")
        print(f"    Sharpe:  {m['sharpe']:.3f}")
        print(f"    Calmar:  {m['calmar']:.3f}")
        for yr, ret in sorted(m['annual_rets'].items()):
            print(f"    {yr}:    {ret:+.2%}")

    # ── 可视化 ────────────────────────────────────────────────────────
    plot_equity(val_eq, test_eq, csi300_val, csi300_test, best_p)
    plot_sensitivity(sens_df, "Val 2024")

    # ── 保存最优参数 ──────────────────────────────────────────────────
    import json
    with open(OUT_DIR / "best_strategy_params.json", "w") as f:
        json.dump({
            "best_params": asdict(best_p),
            "val_metrics": {k: v for k, v in val_m.items()
                            if k not in ('params', 'equity_df', 'annual_rets', 'ic_series')},
            "test_metrics": {k: v for k, v in test_m.items()
                             if k not in ('params', 'equity_df', 'annual_rets', 'ic_series')},
            "default_metrics": {k: v for k, v in default_m.items()
                                 if k not in ('params', 'equity_df', 'annual_rets', 'ic_series')},
            "val_annual_rets": val_m['annual_rets'],
            "test_annual_rets": test_m['annual_rets'],
        }, f, indent=2)
    print(f"\n  最优参数已保存: {OUT_DIR}/best_strategy_params.json")

    print(f"\n  总耗时: {time.time()-t0:.1f}s")
    print("═" * 65)


if __name__ == "__main__":
    main()
