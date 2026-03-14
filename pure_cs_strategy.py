#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯截面选股策略（Pure CS Alpha）
================================

核心改进（相对 index_ma_combined_strategy.py）：
  1. 移除外部择时模型 CSV → 直接从 DB 计算 CSI300 MA 状态
  2. 槽位永不为 0：≥MA60→FULL_SLOTS, MA20-MA60→HALF_SLOTS, <MA20→BEAR_SLOTS
  3. 取消 SLOT_CONFIRM_DAYS 等待期（个股 MA5>MA20 入场过滤替代）
  4. 保留所有个股风控（止损/追踪/MA死叉）

分析依据（2023-2025样本外）：
  - CS top-8 满仓潜力: 2023=+19.6%  2024=+27.9%  2025=+90.5%
  - 时序模型空仓期 CS top-8 仍产生正 alpha（2024均+1.0%/5日，2025均+1.7%/5日）
  - slots=0 期间错过累计收益: 2024=-30.5%  2025=-28.9%

用法:
  python pure_cs_strategy.py
  python pure_cs_strategy.py --slots 10 5 3     # full/half/bear 槽位
  python pure_cs_strategy.py --ma_filter         # 启用个股MA20入场过滤
  python pure_cs_strategy.py --stop 0.07         # 止损线调整
"""

import sys
import warnings
import argparse
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
# 1. 默认配置常量（可由 CLI 覆盖）
# ══════════════════════════════════════════════════════════════════════

DB_PATH        = config.db_path
BACKTEST_START = '20230101'
BACKTEST_END   = '20251231'

# 槽位（每槽权重固定 = 总净值 / FULL_SLOTS，消除级联再平衡）
FULL_SLOTS  = 8    # CSI300 ≥ MA60（牛市）
HALF_SLOTS  = 5    # MA20 ≤ CSI300 < MA60（震荡/中性）
BEAR_SLOTS  = 3    # CSI300 < MA20（熊市），不清零！个股风控负责减仓

# 个股风控
MIN_HOLD_DAYS   = 5      # 持仓满N日后才允许MA死叉触发出场
MA_DEATH_DAYS   = 5      # 连续N日 MA5<MA20 → 死叉出场
STOP_LOSS_ENTRY = 0.08   # 入场价回撤 8% 硬止损
TRAILING_STOP   = 1.00   # 追踪止损（禁用：1.0=永不触发）
TAKE_PROFIT_GAIN  = 1.00
TAKE_PROFIT_TRAIL = 0.10

# 入场过滤
ENTRY_MA20_FILTER = False  # 是否要求个股 close > MA20 才允许买入

# 安全过滤
MIN_LISTED_DAYS = 180
MKTCAP_PCT_CUT  = 20

# 交易成本
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE  = 0.001
TRANSFER_RATE   = 0.00002
SLIPPAGE_RATE   = 0.0001
MIN_COMMISSION  = 5.0
MIN_TRADE_VALUE = 2000.0

INITIAL_CAPITAL = 1_000_000.0

CS_PRED_FILE = ROOT / 'output' / 'csv' / 'xgb_cross_section_predictions.csv'
OUT_DIR      = ROOT / 'output' / 'pure_cs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 2. 工具函数
# ══════════════════════════════════════════════════════════════════════

def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)

def _norm_date(s) -> str:
    return str(s).replace('-', '')


# ══════════════════════════════════════════════════════════════════════
# 3. CSI300 MA 状态 → 槽位（关键改进：不依赖外部 CSV）
# ══════════════════════════════════════════════════════════════════════

def load_csi300_ma_slots(full: int, half: int, bear: int) -> pd.Series:
    """
    从 DB 直接计算 CSI300 每日 MA 状态，映射为槽位数。

    CSI300 ≥ MA60  → full_slots（牛市）
    MA20 ≤ CSI300 < MA60 → half_slots（震荡）
    CSI300 < MA20  → bear_slots（熊市，但永不为 0）

    返回: pd.Series, index=YYYYMMDD str, values=int slots
    """
    print("  从 DB 计算 CSI300 MA 状态...")
    with get_conn() as conn:
        df = conn.execute("""
            SELECT trade_date, close
            FROM index_daily
            WHERE ts_code = '000300.SH'
            ORDER BY trade_date
        """).fetchdf()

    df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')
    df = df.set_index('trade_date').sort_index()

    # 计算 MA（从2022起，保证有足够的预热数据）
    df['ma20'] = df['close'].rolling(20,  min_periods=5).mean()
    df['ma60'] = df['close'].rolling(60,  min_periods=20).mean()

    def _slots(row) -> int:
        c, m20, m60 = row['close'], row['ma20'], row['ma60']
        if pd.isna(m60):
            return half
        if c >= m60:
            return full
        if pd.isna(m20) or c >= m20:
            return half
        return bear

    slots = df.apply(_slots, axis=1)
    slots = slots[slots.index >= '20221201']  # 含预热期

    bull  = (slots == full).sum()
    mid   = (slots == half).sum()
    br    = (slots == bear).sum()
    print(f"  CSI300 MA 状态: 牛市({full}槽)={bull}天  "
          f"震荡({half}槽)={mid}天  熊市({bear}槽)={br}天  "
          f"  空仓率=0%（永不空仓）")
    return slots


# ══════════════════════════════════════════════════════════════════════
# 4. 复用：预测数据 & 价格 & 安全过滤
# ══════════════════════════════════════════════════════════════════════

def load_cs_predictions() -> pd.DataFrame:
    if not CS_PRED_FILE.exists():
        raise FileNotFoundError(
            f"找不到截面选股预测: {CS_PRED_FILE}\n"
            "请先运行 xgboost_cross_section.py 生成预测"
        )
    df = pd.read_csv(CS_PRED_FILE, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    df = df[(df['trade_date'] >= BACKTEST_START) & (df['trade_date'] <= BACKTEST_END)]
    print(f"  截面选股: {len(df):,} 行  "
          f"{df['trade_date'].nunique()} 个调仓日  "
          f"{df['ts_code'].nunique()} 只股票")
    return df[['ts_code', 'trade_date', 'pred']].copy()


def load_price_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("  加载个股日线价格...")
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT trade_date, ts_code, close, vol
            FROM daily_price
            WHERE trade_date >= '20221101'
              AND trade_date <= '{BACKTEST_END}'
              AND ts_code NOT LIKE '8%'
              AND ts_code NOT LIKE '4%'
            ORDER BY trade_date, ts_code
        """).fetchdf()

    df['trade_date'] = df['trade_date'].apply(_norm_date)
    price_pv = df.pivot(index='trade_date', columns='ts_code', values='close')
    vol_pv   = df.pivot(index='trade_date', columns='ts_code', values='vol')
    ma5_pv   = price_pv.rolling(5,  min_periods=3).mean()
    ma20_pv  = price_pv.rolling(20, min_periods=10).mean()

    mask     = price_pv.index >= BACKTEST_START
    print(f"  价格矩阵: {mask.sum()} 天 × {len(price_pv.columns):,} 只股票")
    return price_pv[mask], ma5_pv[mask], ma20_pv[mask], vol_pv[mask]


def load_safety_filters(rebal_dates: List[str]) -> Dict[str, set]:
    print("  计算安全过滤器（ST/新股/小市值）...")
    with get_conn() as conn:
        sb = conn.execute("SELECT ts_code, list_date, name FROM stock_basic").fetchdf()
        dates_sql = "'" + "','".join(rebal_dates) + "'"
        mktcap_df = conn.execute(f"""
            SELECT ts_code, trade_date, total_mv
            FROM daily_basic
            WHERE trade_date IN ({dates_sql}) AND total_mv > 0
        """).fetchdf()

    sb['is_st']     = sb['name'].str.contains('ST', na=False)
    sb['list_date'] = sb['list_date'].apply(_norm_date)
    st_set          = set(sb.loc[sb['is_st'], 'ts_code'])
    list_date_map   = dict(zip(sb['ts_code'], sb['list_date']))
    mktcap_df['trade_date'] = mktcap_df['trade_date'].apply(_norm_date)

    eligible: Dict[str, set] = {}
    for date in rebal_dates:
        sub = mktcap_df[mktcap_df['trade_date'] == date]
        if sub.empty:
            eligible[date] = set()
            continue
        cutoff = np.percentile(sub['total_mv'].values, MKTCAP_PCT_CUT)
        elig = set()
        for _, row in sub.iterrows():
            ts = row['ts_code']
            if ts in st_set or ts.startswith('8') or ts.startswith('4'):
                continue
            if row['total_mv'] <= cutoff:
                continue
            ld = list_date_map.get(ts, '20000101')
            try:
                if (pd.Timestamp(date) - pd.Timestamp(ld)).days < MIN_LISTED_DAYS:
                    continue
            except Exception:
                continue
            elig.add(ts)
        eligible[date] = elig

    avg = np.mean([len(v) for v in eligible.values()]) if eligible else 0
    print(f"  平均合规股票数: {avg:.0f} 只/调仓日")
    return eligible


# ══════════════════════════════════════════════════════════════════════
# 5. 交易成本
# ══════════════════════════════════════════════════════════════════════

def trade_cost(value: float, is_sell: bool) -> float:
    v = abs(value)
    c = max(v * COMMISSION_RATE, MIN_COMMISSION)
    t = v * STAMP_TAX_RATE if is_sell else 0.0
    r = v * TRANSFER_RATE
    s = v * SLIPPAGE_RATE
    return c + t + r + s


# ══════════════════════════════════════════════════════════════════════
# 6. 持仓管理
# ══════════════════════════════════════════════════════════════════════

class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash         = initial_cash
        self.shares:      Dict[str, int]   = {}
        self.entry_price: Dict[str, float] = {}
        self.entry_date:  Dict[str, str]   = {}
        self.peak_price:  Dict[str, float] = {}
        self.hold_days:   Dict[str, int]   = {}
        self.dc_count:    Dict[str, int]   = {}

    def total_value(self, prices: pd.Series) -> float:
        pos_val = sum(
            self.shares.get(ts, 0) * float(prices.get(ts, 0) or 0)
            for ts in list(self.shares)
        )
        return self.cash + pos_val

    def sell(self, ts_code: str, price: float, date: str) -> float:
        sh = self.shares.pop(ts_code, 0)
        if sh == 0 or not np.isfinite(price) or price <= 0:
            for d in [self.entry_price, self.entry_date, self.peak_price,
                      self.hold_days, self.dc_count]:
                d.pop(ts_code, None)
            return 0.0
        val    = sh * price
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
        affordable  = min(target_val, self.cash / cost_factor)
        if affordable < MIN_TRADE_VALUE:
            return False
        sh = int(affordable / price / 100) * 100
        if sh == 0:
            sh = max(1, int(affordable / price))
        val   = sh * price
        total = val + trade_cost(val, is_sell=False)
        if total > self.cash:
            sh = int((self.cash / cost_factor) / price)
            if sh == 0:
                return False
            val   = sh * price
            total = val + trade_cost(val, is_sell=False)
        self.cash -= total
        old_sh = self.shares.get(ts_code, 0)
        old_ep = self.entry_price.get(ts_code, price)
        new_sh = old_sh + sh
        new_ep = (old_ep * old_sh + price * sh) / new_sh
        self.shares[ts_code]      = new_sh
        self.entry_price[ts_code] = new_ep
        self.entry_date[ts_code]  = self.entry_date.get(ts_code, date)
        self.peak_price[ts_code]  = max(self.peak_price.get(ts_code, price), price)
        self.hold_days[ts_code]   = self.hold_days.get(ts_code, 0)
        self.dc_count[ts_code]    = self.dc_count.get(ts_code, 0)
        return True

    def update_hold_days(self):
        for ts in list(self.hold_days):
            self.hold_days[ts] = self.hold_days.get(ts, 0) + 1


# ══════════════════════════════════════════════════════════════════════
# 7. 每日风控
# ══════════════════════════════════════════════════════════════════════

def daily_stop_check(
    portfolio: Portfolio,
    prices:    pd.Series,
    ma5:       pd.Series,
    ma20:      pd.Series,
    date:      str,
) -> List[str]:
    to_sell = []
    for ts in list(portfolio.shares):
        price = prices.get(ts, np.nan)

        # ① 停牌/退市
        if pd.isna(price) or price <= 0:
            to_sell.append(ts)
            continue

        ep   = portfolio.entry_price.get(ts, price)
        peak = portfolio.peak_price.get(ts, price)

        if price > peak:
            portfolio.peak_price[ts] = price
            peak = price

        # ② 硬止损
        if price < ep * (1 - STOP_LOSS_ENTRY):
            to_sell.append(ts)
            continue

        # ③ 追踪止损（TRAILING_STOP=1.0 时永不触发）
        gain  = price / ep - 1
        trail = TAKE_PROFIT_TRAIL if gain >= TAKE_PROFIT_GAIN else TRAILING_STOP
        if price < peak * (1 - trail):
            to_sell.append(ts)
            continue

        # ④ MA 死叉（持仓满 MIN_HOLD_DAYS 后）
        hd = portfolio.hold_days.get(ts, 0)
        if hd >= MIN_HOLD_DAYS:
            m5  = ma5.get(ts, np.nan)
            m20 = ma20.get(ts, np.nan)
            if pd.notna(m5) and pd.notna(m20):
                if m5 < m20:
                    portfolio.dc_count[ts] = portfolio.dc_count.get(ts, 0) + 1
                    if portfolio.dc_count[ts] >= MA_DEATH_DAYS:
                        to_sell.append(ts)
                        continue
                else:
                    portfolio.dc_count[ts] = 0

    return list(set(to_sell))


# ══════════════════════════════════════════════════════════════════════
# 8. 调仓
# ══════════════════════════════════════════════════════════════════════

def rebalance(
    portfolio:     Portfolio,
    target_stocks: List[str],
    n_slots:       int,
    prices:        pd.Series,
    date:          str,
) -> int:
    """n_slots > 0 保证（永不为0）；slot_value 基于 FULL_SLOTS 固定权重"""
    if n_slots == 0:
        return 0

    total_val  = portfolio.total_value(prices)
    slot_value = total_val / FULL_SLOTS   # 权重固定，消除级联再平衡
    target_set  = set(target_stocks[:n_slots])
    current_set = set(portfolio.shares)
    trades = 0

    # 卖出不在目标中 & 满足最短持仓
    for ts in sorted(current_set - target_set):
        hd = portfolio.hold_days.get(ts, 0)
        if hd >= MIN_HOLD_DAYS:
            p = prices.get(ts, np.nan)
            if pd.notna(p) and p > 0:
                portfolio.sell(ts, p, date)
                trades += 1

    # 买入目标中尚未持有（按 pred 分数顺序）
    for ts in target_stocks[:n_slots]:
        if ts in portfolio.shares:
            continue
        p = prices.get(ts, np.nan)
        if pd.isna(p) or p <= 0:
            continue
        if portfolio.buy(ts, p, slot_value, date):
            trades += 1

    return trades


# ══════════════════════════════════════════════════════════════════════
# 9. 回测主循环
# ══════════════════════════════════════════════════════════════════════

def run_backtest(
    price_pv:    pd.DataFrame,
    ma5_pv:      pd.DataFrame,
    ma20_pv:     pd.DataFrame,
    vol_pv:      pd.DataFrame,
    cs_preds:    pd.DataFrame,
    ma_slots:    pd.Series,      # 每日 MA 状态 slots（永不为0）
    eligible:    Dict[str, set],
    rebal_set:   set,
) -> Tuple[pd.DataFrame, List[dict]]:

    print(f"\n[回测] 纯截面选股策略 "
          f"(full={FULL_SLOTS}/half={HALF_SLOTS}/bear={BEAR_SLOTS}槽  "
          f"stop={STOP_LOSS_ENTRY:.0%}  ma20_filter={ENTRY_MA20_FILTER})")

    pf = Portfolio(INITIAL_CAPITAL)
    cs_by_date: Dict[str, pd.DataFrame] = {
        dt: grp for dt, grp in cs_preds.groupby('trade_date')
    }

    all_dates = sorted(d for d in price_pv.index if d >= BACKTEST_START)
    equity:    List[dict] = []
    trade_log: List[dict] = []

    # 槽位日志（用于统计）
    slot_dist = {FULL_SLOTS: 0, HALF_SLOTS: 0, BEAR_SLOTS: 0}

    for i, date in enumerate(all_dates):
        # 当日 MA 槽位（前向填充）
        prior = ma_slots[ma_slots.index <= date]
        slots_today = int(prior.iloc[-1]) if len(prior) > 0 else HALF_SLOTS

        if i % 60 == 0:
            tv = pf.total_value(price_pv.loc[date])
            print(f"  {date}  净值: ¥{tv:>12,.0f}  "
                  f"持仓: {len(pf.shares):2d} 只  slots={slots_today}")

        prices = price_pv.loc[date]
        ma5    = ma5_pv.loc[date]  if date in ma5_pv.index  else pd.Series(dtype=float)
        ma20   = ma20_pv.loc[date] if date in ma20_pv.index else pd.Series(dtype=float)

        # ① 每日风控
        exits = daily_stop_check(pf, prices, ma5, ma20, date)
        for ts in exits:
            p = float(prices.get(ts, pf.entry_price.get(ts, 1.0)) or 1.0)
            cash_rcv = pf.sell(ts, p, date)
            trade_log.append({'date': date, 'ts_code': ts,
                               'action': 'stop_sell', 'price': p, 'cash': cash_rcv})

        # ② 调仓日
        if date in rebal_set:
            slot_dist[slots_today] = slot_dist.get(slots_today, 0) + 1

            elig_set  = eligible.get(date, set())
            vol_today = vol_pv.loc[date] if date in vol_pv.index else pd.Series(dtype=float)
            active    = set(vol_today[vol_today > 0].dropna().index)

            scores_df = cs_by_date.get(date, pd.DataFrame(columns=['ts_code', 'pred']))
            scores_df = scores_df[
                scores_df['ts_code'].isin(elig_set) &
                scores_df['ts_code'].isin(active)
            ].sort_values('pred', ascending=False)

            # 个股 MA20 入场过滤（可选）
            if ENTRY_MA20_FILTER and len(ma20) > 0:
                def _above_ma20(ts_code):
                    p   = prices.get(ts_code, np.nan)
                    m20 = ma20.get(ts_code, np.nan)
                    return pd.notna(p) and pd.notna(m20) and m20 > 0 and p > m20
                scores_df = scores_df[scores_df['ts_code'].apply(_above_ma20)]

            target_stocks = scores_df['ts_code'].tolist()
            if target_stocks:
                n_trades = rebalance(pf, target_stocks, slots_today, prices, date)
                trade_log.append({'date': date, 'ts_code': 'REBAL',
                                   'action': 'rebalance',
                                   'price': slots_today, 'cash': n_trades})

        # ③ 持仓天数 +1
        pf.update_hold_days()

        # ④ 记录净值
        tv = pf.total_value(prices)
        equity.append({'date': date, 'total_value': tv,
                        'cash': pf.cash, 'n_positions': len(pf.shares),
                        'slots': slots_today})

    equity_df = pd.DataFrame(equity).set_index('date')
    equity_df.index = pd.to_datetime(equity_df.index, format='%Y%m%d')

    total_rebal = sum(slot_dist.values())
    print(f"\n  调仓日 slots 分布: "
          f"bull({FULL_SLOTS}槽)={slot_dist.get(FULL_SLOTS,0)}/{total_rebal}  "
          f"half({HALF_SLOTS}槽)={slot_dist.get(HALF_SLOTS,0)}/{total_rebal}  "
          f"bear({BEAR_SLOTS}槽)={slot_dist.get(BEAR_SLOTS,0)}/{total_rebal}")
    print(f"  最终净值: ¥{equity_df['total_value'].iloc[-1]:,.0f}")
    print(f"  平均持仓: {equity_df['n_positions'].mean():.1f} 只")
    return equity_df, trade_log


# ══════════════════════════════════════════════════════════════════════
# 10. 绩效统计
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(equity_df: pd.DataFrame) -> dict:
    vals = equity_df['total_value']
    rets = vals.pct_change().dropna()
    n    = len(rets)
    tpy  = 244
    yrs  = n / tpy

    total_ret  = vals.iloc[-1] / vals.iloc[0] - 1
    annual_ret = (1 + total_ret) ** (1 / yrs) - 1 if yrs > 0 else 0
    annual_vol = rets.std() * np.sqrt(tpy)
    rf_daily   = 0.02 / tpy
    sharpe     = (rets.mean() - rf_daily) / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0
    peak       = vals.expanding().max()
    dd         = (vals - peak) / peak
    max_dd     = float(dd.min())
    calmar     = annual_ret / abs(max_dd) if max_dd != 0 else np.nan
    down       = rets[rets < 0]
    sortino    = (annual_ret - 0.02) / (down.std() * np.sqrt(tpy)) if len(down) > 0 else np.nan
    win_rate   = float((rets > 0).mean())

    annual_rets = {}
    for yr in range(2023, 2027):
        mask = equity_df.index.year == yr
        if mask.sum() < 5:
            continue
        yv = vals[mask]
        annual_rets[str(yr)] = yv.iloc[-1] / yv.iloc[0] - 1

    return dict(
        total_ret=total_ret, annual_ret=annual_ret, annual_vol=annual_vol,
        sharpe=sharpe, sortino=sortino, calmar=calmar,
        max_dd=max_dd, win_rate=win_rate, years=yrs, annual_rets=annual_rets,
    )


def print_metrics(m: dict, equity_df: Optional[pd.DataFrame] = None, label: str = '纯截面选股'):
    print("\n" + "═" * 58)
    print(f"  {label}  回测报告")
    print(f"  选股: XGBoost 截面超额收益  择时: CSI300 MA直接计算")
    print(f"  回测期: {BACKTEST_START} ~ {BACKTEST_END}（严格样本外）")
    print("═" * 58)
    print(f"  总收益率:    {m['total_ret']:>+.2%}")
    print(f"  年化收益率:  {m['annual_ret']:>+.2%}")
    print(f"  年化波动率:  {m['annual_vol']:>.2%}")
    print(f"  夏普比率:    {m['sharpe']:>.3f}")
    print(f"  索提诺比率:  {m['sortino']:>.3f}")
    print(f"  卡玛比率:    {m['calmar']:>.3f}")
    print(f"  最大回撤:    {m['max_dd']:>.2%}")
    print(f"  日胜率:      {m['win_rate']:>.1%}")
    if equity_df is not None:
        avg_pos  = equity_df['n_positions'].mean()
        avg_slot = equity_df['slots'].mean()
        print(f"  平均持仓:    {avg_pos:.1f} 只")
        print(f"  平均槽位:    {avg_slot:.1f} 槽")
    print("  逐年收益:")
    for yr, ret in m['annual_rets'].items():
        bar = '█' * max(0, int(abs(ret) * 100))
        sign = '↑' if ret > 0 else '↓'
        print(f"    {yr}: {ret:>+.2%}  {bar} {sign}")
    print("═" * 58)


# ══════════════════════════════════════════════════════════════════════
# 11. 基准 & 可视化
# ══════════════════════════════════════════════════════════════════════

def load_csi300_benchmark() -> pd.Series:
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT trade_date, close FROM index_daily
            WHERE ts_code = '000300.SH'
              AND trade_date >= '{BACKTEST_START}'
              AND trade_date <= '{BACKTEST_END}'
            ORDER BY trade_date
        """).fetchdf()
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    s = df.set_index('trade_date')['close']
    s.index = pd.to_datetime(s.index, format='%Y%m%d')
    return s / s.iloc[0]


def plot_results(equity_df: pd.DataFrame, m: dict, benchmark: pd.Series,
                 ma_slots: pd.Series, label: str):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), facecolor='#f8f8f8')
    fig.suptitle(
        f'{label}\n'
        f'年化 {m["annual_ret"]:+.2%}  最大回撤 {m["max_dd"]:.2%}  '
        f'夏普 {m["sharpe"]:.3f}  卡玛 {m["calmar"]:.3f}',
        fontsize=12, fontweight='bold', y=0.99,
    )
    nav = equity_df['total_value'] / INITIAL_CAPITAL

    # ① 净值曲线
    ax = axes[0]
    ax.plot(nav.index, nav.values, color='#1a6db4', lw=1.8, label=label)
    bm = benchmark.reindex(nav.index, method='ffill').dropna()
    if len(bm):
        ax.plot(bm.index, bm.values, color='#e05c2a', lw=1.2,
                alpha=0.8, ls='--', label='沪深300')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4)
    ax.set_ylabel('NAV'); ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.25)
    ax.set_title(f"净值曲线  |  逐年: " +
                 "  ".join(f"{yr}: {ret:+.1%}" for yr, ret in m['annual_rets'].items()),
                 fontsize=9)

    # ② 回撤
    ax = axes[1]
    peak = equity_df['total_value'].expanding().max()
    dd   = (equity_df['total_value'] - peak) / peak * 100
    ax.fill_between(dd.index, dd.values, 0, color='#c0392b', alpha=0.35)
    ax.plot(dd.index, dd.values, color='#c0392b', lw=0.8)
    ax.axhline(0, color='gray', ls=':', alpha=0.4)
    ax.set_ylabel('回撤 (%)'); ax.grid(True, alpha=0.25)
    ax.set_title(f"最大回撤: {m['max_dd']:.2%}", fontsize=10)

    # ③ 持仓只数
    ax = axes[2]
    ax.fill_between(equity_df.index, equity_df['n_positions'].values,
                    color='#27ae60', alpha=0.4, step='post')
    ax.set_ylabel('持仓只数'); ax.grid(True, alpha=0.25)
    ax.set_title(f"持仓数量（均值 {equity_df['n_positions'].mean():.1f} 只）", fontsize=10)

    # ④ 槽位（MA 状态）
    ax = axes[3]
    slots_s = equity_df['slots']
    colors = {FULL_SLOTS: '#27ae60', HALF_SLOTS: '#f39c12', BEAR_SLOTS: '#c0392b'}
    labels = {FULL_SLOTS: f'bull({FULL_SLOTS})', HALF_SLOTS: f'half({HALF_SLOTS})',
              BEAR_SLOTS: f'bear({BEAR_SLOTS})'}
    for sv, col in colors.items():
        mask = slots_s == sv
        ax.fill_between(equity_df.index, 0, slots_s.where(mask, 0),
                        color=col, alpha=0.5, step='post', label=labels[sv])
    ax.set_ylabel('Slots'); ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.25)
    ax.set_title('CSI300 MA 状态（从 DB 实时计算，永不为 0）', fontsize=10)

    for ax in axes:
        for yr in range(2023, 2027):
            ax.axvline(pd.Timestamp(f'{yr}-01-01'), color='gray', ls=':', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = OUT_DIR / 'pure_cs_equity.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# 12. 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    global FULL_SLOTS, HALF_SLOTS, BEAR_SLOTS, STOP_LOSS_ENTRY, ENTRY_MA20_FILTER

    parser = argparse.ArgumentParser(description='纯截面选股策略回测')
    parser.add_argument('--slots', nargs=3, type=int, default=[FULL_SLOTS, HALF_SLOTS, BEAR_SLOTS],
                        metavar=('FULL', 'HALF', 'BEAR'),
                        help=f'槽位数 [牛市 震荡 熊市]，默认 {FULL_SLOTS} {HALF_SLOTS} {BEAR_SLOTS}')
    parser.add_argument('--stop',      type=float, default=STOP_LOSS_ENTRY,
                        help=f'硬止损比例（默认 {STOP_LOSS_ENTRY}）')
    parser.add_argument('--ma_filter', action='store_true',
                        help='启用个股 close>MA20 入场过滤')
    args = parser.parse_args()

    FULL_SLOTS  = args.slots[0]
    HALF_SLOTS  = args.slots[1]
    BEAR_SLOTS  = args.slots[2]
    STOP_LOSS_ENTRY    = args.stop
    ENTRY_MA20_FILTER  = args.ma_filter

    label = (f"纯截面选股  ({FULL_SLOTS}/{HALF_SLOTS}/{BEAR_SLOTS}槽  "
             f"stop={STOP_LOSS_ENTRY:.0%}  ma20={ENTRY_MA20_FILTER})")

    print("=" * 62)
    print(f"  {label}")
    print(f"  回测: {BACKTEST_START} ~ {BACKTEST_END}")
    print("=" * 62)

    # 1. CSI300 MA 槽位（直接从 DB，永不为 0）
    print("\n[1/5] 计算 CSI300 MA 状态...")
    ma_slots = load_csi300_ma_slots(FULL_SLOTS, HALF_SLOTS, BEAR_SLOTS)

    # 2. 截面选股预测
    print("\n[2/5] 加载截面选股预测...")
    cs_preds = load_cs_predictions()

    # 3. 价格数据
    print("\n[3/5] 加载个股价格...")
    price_pv, ma5_pv, ma20_pv, vol_pv = load_price_data()

    # 4. 调仓日 & 安全过滤
    all_rebal   = sorted(cs_preds['trade_date'].unique())
    rebal_dates = [d for d in all_rebal if BACKTEST_START <= d <= BACKTEST_END]
    rebal_set   = set(rebal_dates)
    print(f"\n  调仓日: {len(rebal_dates)} 个  ({rebal_dates[0]} ~ {rebal_dates[-1]})")

    print("\n[4/5] 安全过滤...")
    eligible = load_safety_filters(rebal_dates)

    # 5. 回测
    print("\n[5/5] 运行回测...")
    equity_df, trade_log = run_backtest(
        price_pv, ma5_pv, ma20_pv, vol_pv,
        cs_preds, ma_slots, eligible, rebal_set,
    )

    # 绩效 & 输出
    m = compute_metrics(equity_df)
    print_metrics(m, equity_df, label)

    benchmark = load_csi300_benchmark()
    plot_results(equity_df, m, benchmark, ma_slots, label)

    equity_df.to_csv(OUT_DIR / 'pure_cs_equity.csv')
    pd.DataFrame(trade_log).to_csv(OUT_DIR / 'pure_cs_trades.csv', index=False)
    print(f"\n  净值: {OUT_DIR / 'pure_cs_equity.csv'}")
    print(f"  交易: {OUT_DIR / 'pure_cs_trades.csv'}")
    print("\n完成。")


if __name__ == '__main__':
    main()
