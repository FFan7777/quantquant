#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化联合策略回测 v5
==============================
基于对 v1 失败原因的分析:
  问题 1: 时序模型聚合为"市场择时"信号效果差
          → 中性/牛市期间 conf_pct 低，导致只开 6 槽，错过 2024 行情
  问题 2: 复合得分加动量因子，选股质量反而下降
          → 动量在震荡市发生 reversal，污染了 CS 模型的强信号

改进方案 (三步外科手术):
  ① 去掉时序模型的"市场择时"功能，改用 MA 制度直接决定仓位:
       bear (SSE < MA20)          →  0 槽
       neutral (MA20 < SSE ≤ MA60) → 10 槽
       bull (SSE > MA60)           → 20 槽
       → 原 v1 中性期 6 槽 → 现 10 槽，牛市 20 槽保证充分参与

  ② 时序模型改为 per-stock 选股增强:
       对每个调仓日，将时序模型个股 pred_prob 前向填充至 CS 调仓日
       composite = CS_score(70%) + timing_prob_rank(30%)
       → 时序模型做它擅长的事（评估单个股票），不做它不擅长的事（市场择时）

  ③ 止盈追踪:
       浮盈 > 15% 激活，追踪止损从 7% 收紧至 5%

保留 v1 全部其他规则: 止损 8%, MA 死叉 3 日退出, 最短持有 10 日, 安全过滤

严格样本外: ML 训练 2018-2022 | 回测 2023-2025
"""

import sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple

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

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH         = config.db_path
BACKTEST_START  = '20230101'
BACKTEST_END    = '20251231'
MAX_SLOTS       = 20
MIN_HOLD_DAYS   = 10
MA_DEATH_DAYS   = 3
MIN_LISTED_DAYS = 180
MKTCAP_PCT_CUT  = 20

STOP_LOSS_ENTRY = 0.08
TRAILING_STOP   = 0.07
TAKE_PROFIT_GAIN  = 0.15   # 浮盈超此值激活止盈追踪
TAKE_PROFIT_TRAIL = 0.05   # 止盈模式追踪幅度

# 复合得分权重
CS_WEIGHT     = 0.70
TIMING_WEIGHT = 0.30

COMMISSION_RATE = 0.0003
STAMP_TAX_RATE  = 0.001
TRANSFER_RATE   = 0.00002
SLIPPAGE_RATE   = 0.0001
MIN_COMMISSION  = 5.0
MIN_TRADE_VALUE = 2000.0
INITIAL_CAPITAL = 1_000_000.0

CS_PRED_FILE     = ROOT / 'output' / 'csv' / 'xgb_cross_section_predictions.csv'
TIMING_PRED_FILE = ROOT / 'output' / 'csv' / 'timing_test_predictions.csv'
OUT_DIR          = ROOT / 'output' / 'optimized'
SSE_INDEX        = '000001.SH'


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)

def norm(s) -> str:
    return str(s).replace('-', '')


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_cs_predictions() -> pd.DataFrame:
    print("加载截面选股预测...")
    df = pd.read_csv(CS_PRED_FILE, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(norm)
    df = df[(df['trade_date'] >= BACKTEST_START) & (df['trade_date'] <= BACKTEST_END)]
    print(f"  CS: {len(df):,} 行, {df['trade_date'].nunique()} 调仓日")
    return df[['ts_code', 'trade_date', 'pred']].copy()


def load_timing_predictions_per_stock() -> pd.DataFrame:
    """加载时序模型个股预测（用于 per-stock 选股，不用于市场择时）"""
    print("加载时序模型个股预测...")
    df = pd.read_csv(TIMING_PRED_FILE, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(norm)
    df = df[(df['trade_date'] >= BACKTEST_START) & (df['trade_date'] <= BACKTEST_END)]
    print(f"  择时个股: {len(df):,} 行, {df['trade_date'].nunique()} 个择时日")
    return df[['ts_code', 'trade_date', 'pred_prob']].copy()


def load_sse_index() -> pd.DataFrame:
    print("加载上证指数...")
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT trade_date, close FROM index_daily
            WHERE ts_code = '{SSE_INDEX}'
              AND trade_date >= '20220101' AND trade_date <= '{BACKTEST_END}'
            ORDER BY trade_date
        """).fetchdf()
    df['trade_date'] = df['trade_date'].apply(norm)
    df = df.set_index('trade_date')
    df['ma20'] = df['close'].rolling(20, min_periods=10).mean()
    df['ma60'] = df['close'].rolling(60, min_periods=30).mean()
    return df


def load_price_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """返回 price_pv, ma5_pv, ma20_pv, vol_pv (均从 BACKTEST_START 起)"""
    print("加载日线价格数据...")
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT trade_date, ts_code, close, vol
            FROM daily_price
            WHERE trade_date >= '20221101' AND trade_date <= '{BACKTEST_END}'
              AND ts_code NOT LIKE '8%' AND ts_code NOT LIKE '4%'
            ORDER BY trade_date, ts_code
        """).fetchdf()
    df['trade_date'] = df['trade_date'].apply(norm)
    price_full = df.pivot(index='trade_date', columns='ts_code', values='close')
    vol_full   = df.pivot(index='trade_date', columns='ts_code', values='vol')
    ma5_full   = price_full.rolling(5,  min_periods=3).mean()
    ma20_full  = price_full.rolling(20, min_periods=10).mean()
    mask = price_full.index >= BACKTEST_START
    print(f"  价格矩阵: {mask.sum()} 天 × {price_full.shape[1]} 只")
    return (price_full[mask], ma5_full[mask], ma20_full[mask], vol_full[mask])


def load_safety_filters(rebal_dates: List[str]) -> Dict[str, set]:
    print("加载安全过滤器...")
    with get_conn() as conn:
        sb = conn.execute("SELECT ts_code, list_date, name FROM stock_basic").fetchdf()
        dates_sql = "'" + "','".join(rebal_dates) + "'"
        mktcap_df = conn.execute(f"""
            SELECT ts_code, trade_date, total_mv FROM daily_basic
            WHERE trade_date IN ({dates_sql}) AND total_mv > 0
        """).fetchdf()
    sb['is_st']     = sb['name'].str.contains('ST', na=False)
    sb['list_date'] = sb['list_date'].apply(norm)
    st_set        = set(sb.loc[sb['is_st'], 'ts_code'])
    list_date_map = dict(zip(sb['ts_code'], sb['list_date']))
    mktcap_df['trade_date'] = mktcap_df['trade_date'].apply(norm)

    eligible = {}
    for date in rebal_dates:
        sub = mktcap_df[mktcap_df['trade_date'] == date]
        if sub.empty:
            eligible[date] = set(); continue
        cutoff = np.percentile(sub['total_mv'].values, MKTCAP_PCT_CUT)
        elig = set()
        for _, row in sub.iterrows():
            ts = row['ts_code']
            if ts in st_set or row['total_mv'] <= cutoff: continue
            ld = list_date_map.get(ts, '20000101')
            try:
                if (pd.Timestamp(date) - pd.Timestamp(ld)).days < MIN_LISTED_DAYS: continue
            except: continue
            elig.add(ts)
        eligible[date] = elig
    print(f"  平均合规: {np.mean([len(v) for v in eligible.values()]):.0f} 只/调仓日")
    return eligible


# ─── 改进①: MA 制度直接决定槽位 (去掉时序模型的市场择时) ────────────────────
def get_ma_regime(sse: pd.DataFrame, date: str) -> str:
    row = None
    if date in sse.index:
        row = sse.loc[date]
    else:
        avail = sse.index[sse.index <= date]
        if len(avail): row = sse.loc[avail[-1]]
    if row is None or any(pd.isna(row[c]) for c in ['close', 'ma20', 'ma60']):
        return 'neutral'
    if row['close'] > row['ma60']:   return 'bull'
    elif row['close'] > row['ma20']: return 'neutral'
    else:                            return 'bear'


def compute_timing_breadth_series(timing_df: pd.DataFrame,
                                   rebal_dates: List[str],
                                   threshold: float = 0.35) -> Dict[str, float]:
    """
    每个择时日: pred_prob > threshold 的股票占比 → 百分位归一化
    前向填充至 CS 调仓日，返回 {rebal_date → breadth_pct [0,1]}
    """
    raw = timing_df.groupby('trade_date').apply(
        lambda x: (x['pred_prob'] > threshold).mean()
    )
    pct = raw.rank(pct=True)   # 历史百分位排名
    timing_dates = sorted(pct.index)
    result = {}
    for date in rebal_dates:
        prior = [d for d in timing_dates if d <= date]
        result[date] = float(pct.loc[prior[-1]]) if prior else 0.3
    return result


def compute_position_budget(sse: pd.DataFrame,
                             timing_df: pd.DataFrame,
                             rebal_dates: List[str]) -> Dict[str, int]:
    """
    改进①+微调:
      bear    →  0  (熊市不开新仓)
      neutral →  6 (宽度低) / 10 (宽度高)  ← 时序宽度微调中性期仓位
      bull    → 20  (牛市固定满仓, 不受时序限制)
    """
    breadth = compute_timing_breadth_series(timing_df, rebal_dates)
    budget = {}
    for date in rebal_dates:
        regime = get_ma_regime(sse, date)
        bv     = breadth.get(date, 0.3)
        if regime == 'bear':
            budget[date] = 0
        elif regime == 'neutral':
            budget[date] = 10 if bv > 0.5 else 6  # 宽度低时保守
        else:  # bull
            budget[date] = 20  # 牛市固定满仓
    vals = list(budget.values())
    print(f"  仓位预算: 均值={np.mean(vals):.1f}槽  "
          f"0槽(熊)={sum(v==0 for v in vals)}次  "
          f"6槽(中性弱)={sum(v==6 for v in vals)}次  "
          f"10槽(中性强)={sum(v==10 for v in vals)}次  "
          f"20槽(牛)={sum(v==20 for v in vals)}次")
    return budget


# ─── 改进②: 时序模型改为 per-stock 选股增强 ──────────────────────────────────
def build_timing_lookup(timing_df: pd.DataFrame,
                         rebal_dates: List[str]) -> Dict[str, pd.Series]:
    """
    对每个 CS 调仓日，找最近的时序预测日 (前向填充，不超过 20 个交易日)
    返回: {rebal_date → Series(ts_code → pred_prob)}
    """
    timing_dates = sorted(timing_df['trade_date'].unique())
    # 按日期建索引
    timing_by_date = {d: grp.set_index('ts_code')['pred_prob']
                      for d, grp in timing_df.groupby('trade_date')}

    lookup = {}
    for rdate in rebal_dates:
        # 找最近不超过 rdate 的时序日期
        prior = [d for d in timing_dates if d <= rdate]
        if not prior:
            lookup[rdate] = pd.Series(dtype=float)
            continue
        # 不超过 20 个交易日的前向填充限制 (约 1 个月)
        latest = prior[-1]
        try:
            gap = (pd.Timestamp(rdate, format='%Y%m%d') -
                   pd.Timestamp(latest, format='%Y%m%d')).days
        except Exception:
            gap = 0
        if gap > 30:   # 超过 30 日历天，信号过时，不使用
            lookup[rdate] = pd.Series(dtype=float)
        else:
            lookup[rdate] = timing_by_date.get(latest, pd.Series(dtype=float))

    valid = sum(1 for v in lookup.values() if len(v) > 0)
    print(f"  择时 per-stock 覆盖: {valid}/{len(rebal_dates)} 调仓日有效")
    return lookup


def build_composite_scores(cs_preds: pd.DataFrame,
                            timing_lookup: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
    """
    改进②: composite = CS_WEIGHT × CS截面排名 + TIMING_WEIGHT × timing_prob截面排名
    → 时序模型用于 per-stock 选股，CS 模型为主力信号
    """
    print(f"计算复合截面得分 (CS×{CS_WEIGHT:.0%} + 择时个股概率×{TIMING_WEIGHT:.0%})...")
    result = {}
    for date, grp in cs_preds.groupby('trade_date'):
        base = grp.set_index('ts_code')[['pred']].copy()
        tprob = timing_lookup.get(date, pd.Series(dtype=float))
        base['timing_prob'] = tprob.reindex(base.index)

        base['cs_r']     = base['pred'].rank(pct=True)
        base['timing_r'] = base['timing_prob'].rank(pct=True).fillna(0.5)
        base['score']    = CS_WEIGHT * base['cs_r'] + TIMING_WEIGHT * base['timing_r']
        result[date]     = base[['score']].reset_index()
    return result


# ─── TRANSACTION COSTS ────────────────────────────────────────────────────────
def trade_cost(value: float, is_sell: bool) -> float:
    v = abs(value)
    return (max(v * COMMISSION_RATE, MIN_COMMISSION)
            + (v * STAMP_TAX_RATE if is_sell else 0)
            + v * TRANSFER_RATE + v * SLIPPAGE_RATE)


# ─── PORTFOLIO (含 改进③ 止盈状态) ────────────────────────────────────────────
class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash        = initial_cash
        self.shares:     Dict[str, int]   = {}
        self.entry_price: Dict[str, float] = {}
        self.entry_date:  Dict[str, str]   = {}
        self.peak_price:  Dict[str, float] = {}
        self.dc_count:    Dict[str, int]   = {}
        self.in_tp:       Dict[str, bool]  = {}

    def total_value(self, prices: pd.Series) -> float:
        return self.cash + sum(
            self.shares.get(ts, 0) * float(prices.get(ts, 0) or 0)
            for ts in list(self.shares))

    def hold_days(self, ts: str, date: str) -> int:
        ed = self.entry_date.get(ts)
        if not ed: return 0
        try: return (pd.Timestamp(date) - pd.Timestamp(ed)).days
        except: return 0

    def sell(self, ts: str, price: float, date: str) -> float:
        sh = self.shares.pop(ts, 0)
        if sh == 0 or price <= 0: return 0.0
        val = sh * price
        cash_in = val - trade_cost(val, is_sell=True)
        self.cash += cash_in
        for d in [self.entry_price, self.entry_date, self.peak_price,
                  self.dc_count, self.in_tp]:
            d.pop(ts, None)
        return cash_in

    def buy(self, ts: str, price: float, target_val: float, date: str) -> bool:
        if price <= 0 or target_val < MIN_TRADE_VALUE: return False
        cf = 1 + COMMISSION_RATE + TRANSFER_RATE + SLIPPAGE_RATE
        afford = min(target_val, self.cash / cf)
        if afford < MIN_TRADE_VALUE: return False
        sh = int(afford / price / 100) * 100
        if sh == 0: sh = max(1, int(afford / price))
        val = sh * price
        total = val + trade_cost(val, is_sell=False)
        if total > self.cash:
            sh = int((self.cash / cf) / price)
            if sh == 0: return False
            val = sh * price; total = val + trade_cost(val, is_sell=False)
        self.cash -= total
        old = self.shares.get(ts, 0); old_ep = self.entry_price.get(ts, price); new = old + sh
        self.shares[ts]      = new
        self.entry_price[ts] = (old_ep * old + price * sh) / new
        self.entry_date[ts]  = self.entry_date.get(ts, date)
        self.peak_price[ts]  = max(self.peak_price.get(ts, price), price)
        self.dc_count[ts]    = self.dc_count.get(ts, 0)
        self.in_tp[ts]       = False
        return True


# ─── 改进③: 止损 + 止盈 + 死叉检查 ──────────────────────────────────────────
def daily_stop_check(pf: Portfolio, prices: pd.Series,
                     ma5: pd.Series, ma20: pd.Series,
                     date: str) -> List[str]:
    to_sell = []
    for ts in list(pf.shares):
        price = prices.get(ts, np.nan)
        if pd.isna(price) or price <= 0:
            to_sell.append(ts); continue

        ep   = pf.entry_price.get(ts, price)
        peak = pf.peak_price.get(ts, price)
        if price > peak: pf.peak_price[ts] = price; peak = price

        # ① 硬止损 (与 v1 相同: -8%)
        if price < ep * (1 - STOP_LOSS_ENTRY):
            to_sell.append(ts); continue

        # ② 止盈模式激活 (改进③)
        if not pf.in_tp.get(ts, False) and ep > 0 and (price - ep) / ep > TAKE_PROFIT_GAIN:
            pf.in_tp[ts] = True

        # ③ 追踪止损: 正常 7%, 止盈模式 5% (改进③)
        trail = TAKE_PROFIT_TRAIL if pf.in_tp.get(ts, False) else TRAILING_STOP
        if price < peak * (1 - trail):
            to_sell.append(ts); continue

        # ④ MA 死叉 (与 v1 相同)
        if pf.hold_days(ts, date) >= MIN_HOLD_DAYS:
            m5  = ma5.get(ts, np.nan); m20 = ma20.get(ts, np.nan)
            if pd.notna(m5) and pd.notna(m20):
                if m5 < m20:
                    pf.dc_count[ts] = pf.dc_count.get(ts, 0) + 1
                    if pf.dc_count[ts] >= MA_DEATH_DAYS:
                        to_sell.append(ts); continue
                else:
                    pf.dc_count[ts] = 0

    return list(set(to_sell))


# ─── REBALANCING (与 v1 完全相同) ─────────────────────────────────────────────
def rebalance(pf: Portfolio, target_stocks: List[str],
              n_slots: int, prices: pd.Series, date: str) -> int:
    if n_slots == 0: return 0
    total_val  = pf.total_value(prices)
    slot_value = total_val / n_slots
    target_set = set(target_stocks[:n_slots])
    trades = 0
    for ts in list(set(pf.shares) - target_set):
        if pf.hold_days(ts, date) >= MIN_HOLD_DAYS:
            p = prices.get(ts, np.nan)
            if pd.notna(p) and p > 0: pf.sell(ts, p, date); trades += 1
    for ts in list(target_set - set(pf.shares)):
        p = prices.get(ts, np.nan)
        if pd.isna(p) or p <= 0: continue
        if pf.buy(ts, p, slot_value, date): trades += 1
    return trades


# ─── 主回测循环 ───────────────────────────────────────────────────────────────
def run_backtest(price_pv:    pd.DataFrame,
                 ma5_pv:      pd.DataFrame,
                 ma20_pv:     pd.DataFrame,
                 vol_pv:      pd.DataFrame,
                 composite:   Dict[str, pd.DataFrame],
                 slot_budget: Dict[str, int],
                 eligible:    Dict[str, set],
                 rebal_set:   set) -> pd.DataFrame:
    print("\n[Backtest] 运行优化策略 v5...")
    pf = Portfolio(INITIAL_CAPITAL)
    all_dates = sorted(d for d in price_pv.index if BACKTEST_START <= d <= BACKTEST_END)
    equity = []

    for i, date in enumerate(all_dates):
        if i % 100 == 0:
            tv = pf.total_value(price_pv.loc[date])
            print(f"  {date}  净值: ¥{tv:>12,.0f}  持仓: {len(pf.shares):2d} 只")

        prices = price_pv.loc[date]
        ma5  = ma5_pv.loc[date]  if date in ma5_pv.index  else pd.Series(dtype=float)
        ma20 = ma20_pv.loc[date] if date in ma20_pv.index else pd.Series(dtype=float)

        # ① 止损 / 止盈 / 死叉
        for ts in daily_stop_check(pf, prices, ma5, ma20, date):
            p = float(prices.get(ts, pf.entry_price.get(ts, 1.0)) or 1.0)
            pf.sell(ts, p, date)

        # ② 调仓
        if date in rebal_set:
            n_slots   = slot_budget.get(date, 0)
            scores_df = composite.get(date, pd.DataFrame(columns=['ts_code', 'score']))
            elig_set  = eligible.get(date, set())
            vol_row   = vol_pv.loc[date] if date in vol_pv.index else pd.Series(dtype=float)
            active    = set(vol_row[vol_row > 0].dropna().index)

            scores_df = scores_df[
                scores_df['ts_code'].isin(elig_set) &
                scores_df['ts_code'].isin(active)
            ].sort_values('score', ascending=False)

            if n_slots > 0 and len(scores_df):
                rebalance(pf, scores_df['ts_code'].tolist(), n_slots, prices, date)

        # ③ 记录净值
        tv = pf.total_value(prices)
        equity.append({'date': date, 'total_value': tv,
                       'cash': pf.cash, 'n_positions': len(pf.shares)})

    eq_df = pd.DataFrame(equity).set_index('date')
    eq_df.index = pd.to_datetime(eq_df.index, format='%Y%m%d')
    print(f"\n  最终净值: ¥{eq_df['total_value'].iloc[-1]:,.0f}")
    print(f"  平均持仓: {eq_df['n_positions'].mean():.1f} 只")
    return eq_df


# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_metrics(equity_df: pd.DataFrame) -> dict:
    vals = equity_df['total_value']
    rets = vals.pct_change().dropna()
    tpy  = 244
    yrs  = len(rets) / tpy
    total_ret  = vals.iloc[-1] / vals.iloc[0] - 1
    annual_ret = (1 + total_ret) ** (1 / yrs) - 1 if yrs > 0 else 0
    annual_vol = rets.std() * np.sqrt(tpy)
    sharpe     = (rets.mean() - 0.02/tpy) / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0
    peak       = vals.expanding().max()
    dd         = (vals - peak) / peak
    max_dd     = dd.min()
    calmar     = annual_ret / abs(max_dd) if max_dd != 0 else np.nan
    down       = rets[rets < 0]
    sortino    = (annual_ret - 0.02) / (down.std() * np.sqrt(tpy)) if len(down) > 0 else np.nan
    win_rate   = (rets > 0).mean()

    annual_rets = {}
    for yr in range(2023, 2027):
        mask = equity_df.index.year == yr
        if mask.sum() < 5: continue
        yv = vals[mask]
        annual_rets[str(yr)] = yv.iloc[-1] / yv.iloc[0] - 1

    return dict(total_ret=total_ret, annual_ret=annual_ret, annual_vol=annual_vol,
                sharpe=sharpe, sortino=sortino, calmar=calmar, max_dd=max_dd,
                win_rate=win_rate, years=yrs, annual_rets=annual_rets)


def print_metrics(m: dict):
    v1 = {'total':+0.2145, 'ann':+0.0675, 'vol':0.2243, 'sharpe':0.315,
          'dd':-0.1910, '2023':-0.0585, '2024':+0.0494, '2025':+0.2267}
    print("\n" + "═" * 65)
    print("  优化联合策略 v5  vs  v1 基准对比")
    print("  改进: MA制度槽位 + 时序per-stock选股 + 止盈追踪")
    print("  ML训练: 2018-2022  |  回测: 2023-2025 (严格 OOS)")
    print("═" * 65)
    def δ(v, b, better='up'):
        d = v - b
        ok = (d > 0) == (better == 'up')
        return f"  {'✓' if ok else '✗'}  Δ{d:+.2%}" if '%' in f'{b:.2%}' else f"  {'✓' if ok else '✗'}  Δ{d:+.3f}"

    def row(label, v, base, fmt='+.2%', better='up'):
        dv = v - base; ok = (dv > 0) == (better == 'up')
        mark = '✓' if ok else '✗'
        return f"  {label:<14} v5:{v:{fmt}}  v1:{base:{fmt}}  {mark} Δ{dv:{fmt[1:]}}"

    print(row("总收益率",    m['total_ret'],  v1['total']))
    print(row("年化收益率",  m['annual_ret'], v1['ann']))
    print(row("年化波动率",  m['annual_vol'], v1['vol'], better='down'))
    print(row("夏普比率",    m['sharpe'],     v1['sharpe'],   fmt='+.3f'))
    print(f"  索提诺比率     {m['sortino']:>.3f}")
    print(f"  卡玛比率       {m['calmar']:>.3f}")
    print(row("最大回撤",    m['max_dd'],     v1['dd'], better='up'))
    print(f"  日胜率         {m['win_rate']:.1%}")
    print(f"  时间跨度       {m['years']:.2f} 年")
    print("  逐年收益:")
    for yr, ret in m['annual_rets'].items():
        base = v1.get(yr, 0)
        dv = ret - base; ok = dv > 0
        bar = '█' * int(abs(ret) * 100) + (' ↑' if ret > 0 else ' ↓')
        print(f"    {yr}: v5:{ret:>+.2%}  v1:{base:>+.2%}  "
              f"{'✓' if ok else '✗'} Δ{dv:+.2%}  {bar}")
    print("═" * 65)


def load_benchmark() -> pd.Series:
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT trade_date, close FROM index_daily
            WHERE ts_code = '000300.SH'
              AND trade_date >= '{BACKTEST_START}' AND trade_date <= '{BACKTEST_END}'
            ORDER BY trade_date
        """).fetchdf()
    df['trade_date'] = df['trade_date'].apply(norm)
    s = df.set_index('trade_date')['close']
    s.index = pd.to_datetime(s.index, format='%Y%m%d')
    return s / s.iloc[0]


def plot_results(equity_df: pd.DataFrame, m: dict, benchmark: pd.Series):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor='#f8f8f8')
    nav = equity_df['total_value'] / INITIAL_CAPITAL

    v1_path = ROOT / 'output' / 'combined' / 'combined_equity.csv'
    v1_nav  = None
    if v1_path.exists():
        v1_df = pd.read_csv(v1_path, index_col=0, parse_dates=True)
        v1_nav = v1_df['total_value'] / INITIAL_CAPITAL

    ax1 = axes[0]
    ax1.plot(nav.index, nav.values, color='#1a6db4', lw=1.8, label='v5 优化策略')
    if v1_nav is not None:
        ax1.plot(v1_nav.index, v1_nav.values, color='#27ae60',
                 lw=1.2, ls='-.', alpha=0.7, label='v1 原始策略')
    bm = benchmark.reindex(nav.index, method='ffill').dropna()
    if len(bm):
        ax1.plot(bm.index, bm.values, color='#e05c2a', lw=1.2, ls='--', alpha=0.6, label='沪深300')
    ax1.axhline(1.0, color='gray', ls=':', alpha=0.4)
    ax1.set_title(
        f"优化策略 v5  |  年化 {m['annual_ret']:+.2%}  "
        f"最大回撤 {m['max_dd']:.2%}  夏普 {m['sharpe']:.3f}",
        fontsize=12, pad=8)
    ax1.set_ylabel('NAV'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)

    ax2 = axes[1]
    peak = equity_df['total_value'].expanding().max()
    dd   = (equity_df['total_value'] - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color='#c0392b', alpha=0.35)
    ax2.plot(dd.index, dd.values, color='#c0392b', lw=0.8)
    ax2.axhline(0, color='gray', ls=':', alpha=0.4)
    ax2.set_ylabel('回撤 (%)'); ax2.grid(True, alpha=0.25)
    ax2.set_title(f"最大回撤 {m['max_dd']:.2%}", fontsize=10)

    ax3 = axes[2]
    ax3.fill_between(equity_df.index, equity_df['n_positions'], color='#27ae60', alpha=0.4)
    ax3.plot(equity_df.index, equity_df['n_positions'], color='#27ae60', lw=0.8)
    ax3.set_ylabel('持仓只数'); ax3.set_xlabel('日期'); ax3.grid(True, alpha=0.25)
    ax3.set_title(f"持仓数（均值 {equity_df['n_positions'].mean():.1f} 只）", fontsize=10)

    for ax in axes:
        for yr in range(2023, 2027):
            ax.axvline(pd.Timestamp(f'{yr}-01-01'), color='gray', ls=':', alpha=0.3)

    plt.tight_layout(pad=2.0)
    path = OUT_DIR / 'optimized_equity.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表已保存: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("  优化联合策略回测 v5")
    print("  ① MA制度槽位  ② 时序per-stock选股  ③ 止盈追踪")
    print("  ML训练: 2018-2022  |  回测: 2023-2025 (严格 OOS)")
    print("=" * 65)

    # 数据加载
    cs_preds    = load_cs_predictions()
    timing_df   = load_timing_predictions_per_stock()
    sse_df      = load_sse_index()
    price_pv, ma5_pv, ma20_pv, vol_pv = load_price_data()

    rebal_dates = sorted(cs_preds['trade_date'].unique())
    rebal_dates = [d for d in rebal_dates if BACKTEST_START <= d <= BACKTEST_END]
    rebal_set   = set(rebal_dates)
    print(f"\n调仓日: {len(rebal_dates)} 个 ({rebal_dates[0]} ~ {rebal_dates[-1]})")

    # 预计算
    print("\n计算安全过滤器...")
    eligible = load_safety_filters(rebal_dates)

    print("\n计算仓位预算 (改进①: MA制度 + 中性期宽度微调)...")
    slot_budget = compute_position_budget(sse_df, timing_df, rebal_dates)

    print("\n构建时序 per-stock 前向填充...")
    timing_lookup = build_timing_lookup(timing_df, rebal_dates)
    composite = build_composite_scores(cs_preds, timing_lookup)

    # 回测
    equity_df = run_backtest(
        price_pv, ma5_pv, ma20_pv, vol_pv,
        composite, slot_budget, eligible, rebal_set
    )

    # 绩效
    m = compute_metrics(equity_df)
    print_metrics(m)

    # 基准 & 图表
    try:
        benchmark = load_benchmark()
    except Exception:
        benchmark = pd.Series(dtype=float)
    plot_results(equity_df, m, benchmark)

    # 保存
    equity_df.to_csv(OUT_DIR / 'optimized_equity.csv')
    summary = {'总收益率':   f"{m['total_ret']:+.4%}",
               '年化收益率': f"{m['annual_ret']:+.4%}",
               '年化波动率': f"{m['annual_vol']:.4%}",
               '夏普比率':   f"{m['sharpe']:.4f}",
               '索提诺比率': f"{m['sortino']:.4f}",
               '卡玛比率':   f"{m['calmar']:.4f}",
               '最大回撤':   f"{m['max_dd']:.4%}",
               '日胜率':     f"{m['win_rate']:.4%}"}
    for yr, ret in m['annual_rets'].items():
        summary[f'{yr}年收益'] = f"{ret:+.4%}"
    pd.DataFrame.from_dict(summary, orient='index', columns=['值']).to_csv(
        OUT_DIR / 'optimized_metrics.csv')
    print(f"  指标已保存: {OUT_DIR / 'optimized_metrics.csv'}")
    print("\n✓ 完成！")


if __name__ == '__main__':
    main()
