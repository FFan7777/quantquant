#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined ML Strategy Backtest
==============================
截面选股模型 + 时序择时模型 + 规则兜底

架构:
  截面选股 XGBoost (已训练 2018-2022, 预测 2023-2025)
    → 预测个股 10 日超额收益百分位 → 每 5 日选 top-N 持仓

  时序择时 XGBoost (已训练 2018-2022, 预测 2023-2025)
    → 聚合市场整体买入信心 → 控制最大持仓槽位 (0-20)

  规则兜底 (每日执行):
    → 上证 MA20/MA60 三档仓位硬约束
    → 个股止损: -8% from entry (硬止损)
    → 个股止损: -7% from peak (追踪止损)
    → MA5 < MA20 连续 3 天 (持有满 10 天后触发)
    → 排除: ST / 停牌 / 上市不满 180 天 / 市值后 20%

严格样本外: 训练 2018-2022 | 回测 2023-2025 (无重叠)
"""

import os
import sys
import warnings
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

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH = config.db_path

BACKTEST_START = '20230101'
BACKTEST_END   = '20251231'

# Position management
MAX_SLOTS      = 20       # 最大槽位 (每槽 = 总净值 / MAX_SLOTS)
REBAL_FREQ     = 5        # 每 5 个交易日调仓一次
MIN_HOLD_DAYS  = 10       # 最短持有天数，达到后才允许 MA 触发退出
MA_DEATH_DAYS  = 3        # MA5 < MA20 连续天数达到此值 → 退出

# Risk controls
STOP_LOSS_ENTRY = 0.08    # 距入场价回撤 8% → 硬止损
TRAILING_STOP   = 0.07    # 距最高点回撤 7% → 追踪止损

# Safety filters
MIN_LISTED_DAYS = 180     # 上市不满 180 天排除
MKTCAP_PCT_CUT  = 20      # 排除市值后 20%

# Transaction costs (A 股标准)
COMMISSION_RATE = 0.0003  # 万三双边
STAMP_TAX_RATE  = 0.001   # 千一印花税 (仅卖出)
TRANSFER_RATE   = 0.00002 # 过户费 0.002%
SLIPPAGE_RATE   = 0.0001  # 滑点 0.01% (单边)
MIN_COMMISSION  = 5.0     # 最低佣金 5 元
MIN_TRADE_VALUE = 2000.0  # 忽略小于 2000 元的漂移调整

INITIAL_CAPITAL = 1_000_000.0  # 初始资金 100 万

# Prediction file paths (strictly OOS: trained 2018-2022)
CS_PRED_FILE     = ROOT / 'output' / 'csv' / 'xgb_cross_section_predictions.csv'
TIMING_PRED_FILE = ROOT / 'output' / 'csv' / 'timing_test_predictions.csv'
OUT_DIR          = ROOT / 'output' / 'combined'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SSE_INDEX = '000001.SH'  # 上证综合指数


# ── DATABASE ──────────────────────────────────────────────────────────────────
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def _norm_date(s: str) -> str:
    """'2023-02-07' → '20230207'，已是 YYYYMMDD 则不变"""
    return str(s).replace('-', '')


# ── LOAD PREDICTIONS ─────────────────────────────────────────────────────────
def load_cs_predictions() -> pd.DataFrame:
    """加载截面选股预测（2023-2025）"""
    print("加载截面选股预测...")
    df = pd.read_csv(CS_PRED_FILE, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    df = df[(df['trade_date'] >= BACKTEST_START.replace('-', '')) &
            (df['trade_date'] <= BACKTEST_END.replace('-', ''))]
    print(f"  CS 预测: {len(df):,} 行, {df['trade_date'].nunique()} 个调仓日, "
          f"{df['ts_code'].nunique()} 只股票")
    return df[['ts_code', 'trade_date', 'pred']].copy()


def load_timing_predictions() -> pd.DataFrame:
    """加载时序择时预测（2023-2025），聚合为市场信心指标"""
    print("加载时序择时预测...")
    df = pd.read_csv(TIMING_PRED_FILE, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    df = df[(df['trade_date'] >= BACKTEST_START.replace('-', '')) &
            (df['trade_date'] <= BACKTEST_END.replace('-', ''))]

    # 聚合：每日跨股票的平均预测概率 → 市场整体买入信心
    agg = df.groupby('trade_date')['pred_prob'].agg(
        mean_prob='mean',
        pct_above_median=lambda x: (x > x.median()).mean()  # 高于当日中位数的比例
    ).reset_index()

    # 百分位排名归一化 → [0, 1]，对自身历史分布做排名
    agg['conf_pct'] = agg['mean_prob'].rank(pct=True)

    print(f"  择时信号: {len(agg)} 个日期, mean_prob ∈ "
          f"[{agg['mean_prob'].min():.3f}, {agg['mean_prob'].max():.3f}]")
    return agg.set_index('trade_date')


# ── SSE INDEX & MARKET REGIME ─────────────────────────────────────────────────
def load_sse_index() -> pd.DataFrame:
    """加载上证指数（含 MA20、MA60）"""
    print("加载上证指数...")
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT trade_date, close
            FROM index_daily
            WHERE ts_code = '{SSE_INDEX}'
              AND trade_date >= '20220101'
              AND trade_date <= '{BACKTEST_END}'
            ORDER BY trade_date
        """).fetchdf()
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    df = df.set_index('trade_date')
    df['ma20'] = df['close'].rolling(20, min_periods=10).mean()
    df['ma60'] = df['close'].rolling(60, min_periods=30).mean()
    print(f"  上证: {len(df)} 天")
    return df


def get_ma_regime(sse: pd.DataFrame, date: str) -> str:
    """
    返回当日市场 MA 状态:
      'bull'    → SSE > MA60  → 允许满仓 (up to MAX_SLOTS)
      'neutral' → MA20 < SSE ≤ MA60 → 允许半仓 (up to MAX_SLOTS//2)
      'bear'    → SSE < MA20  → 禁止新开仓 (return 0 slots)
    """
    row = sse.reindex([date]).iloc[0] if date in sse.index else None
    if row is None:
        # 找最近可用日期
        avail = sse.index[sse.index <= date]
        if len(avail) == 0:
            return 'neutral'
        row = sse.loc[avail[-1]]

    close, ma20, ma60 = row['close'], row['ma20'], row['ma60']
    if any(pd.isna(v) for v in [close, ma20, ma60]):
        return 'neutral'
    if close > ma60:
        return 'bull'
    elif close > ma20:
        return 'neutral'
    else:
        return 'bear'


def compute_position_budget(timing_agg: pd.DataFrame,
                            sse: pd.DataFrame,
                            rebal_dates: List[str]) -> Dict[str, int]:
    """
    每个调仓日的最大新开仓槽位数。

    逻辑:
      ML 信号 (时序模型聚合 conf_pct) → ml_slots
        conf_pct > 0.75  → 20 槽 (强买)
        conf_pct > 0.50  → 15 槽 (偏多)
        conf_pct > 0.25  → 10 槽 (中性)
        else             →  6 槽 (偏空)

      MA 规则覆盖 (硬约束):
        bear   → 0 槽  (禁止新开仓，现有持仓由止损/MA自然退出)
        neutral → min(ml_slots, MAX_SLOTS // 2)
        bull   → ml_slots
    """
    # 前向填充时序信号到所有调仓日
    conf_series = timing_agg['conf_pct'].reindex(rebal_dates).ffill().fillna(0.5)

    budget = {}
    for date in rebal_dates:
        conf = conf_series.get(date, 0.5)

        # ML 信号 → 槽位
        if conf > 0.75:
            ml_slots = MAX_SLOTS
        elif conf > 0.50:
            ml_slots = 15
        elif conf > 0.25:
            ml_slots = 10
        else:
            ml_slots = 6

        # MA 规则硬约束
        regime = get_ma_regime(sse, date)
        if regime == 'bear':
            budget[date] = 0
        elif regime == 'neutral':
            budget[date] = min(ml_slots, MAX_SLOTS // 2)
        else:
            budget[date] = ml_slots

    # 汇报
    vals = list(budget.values())
    print(f"  仓位预算: 均值={np.mean(vals):.1f}槽, "
          f"0槽(熊市)={sum(v == 0 for v in vals)}次, "
          f"满仓(20槽)={sum(v == MAX_SLOTS for v in vals)}次")
    return budget


# ── DAILY PRICE DATA ──────────────────────────────────────────────────────────
def load_price_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载回测期日线价格（含 2022-11 预热期用于 MA 计算）。
    返回: (price_pivot, ma5_pivot, ma20_pivot, vol_pivot)  ── date × ts_code
    """
    print("加载日线价格数据...")
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

    ma5_pv  = price_pv.rolling(5,  min_periods=3).mean()
    ma20_pv = price_pv.rolling(20, min_periods=10).mean()

    # 裁剪到回测期
    mask = price_pv.index >= BACKTEST_START
    price_pv = price_pv[mask]
    vol_pv   = vol_pv[mask]
    ma5_pv   = ma5_pv[mask]
    ma20_pv  = ma20_pv[mask]

    print(f"  价格矩阵: {len(price_pv)} 天 × {len(price_pv.columns)} 只股票")
    return price_pv, ma5_pv, ma20_pv, vol_pv


# ── SAFETY FILTERS ────────────────────────────────────────────────────────────
def load_safety_filters(rebal_dates: List[str]) -> Dict[str, set]:
    """
    每个调仓日返回合规股票集合:
      排除: ST / 上市不满 180 天 / 市值后 20% / 北交所(8*/4*)
    """
    print("加载安全过滤器...")
    with get_conn() as conn:
        # 静态信息 (ST 状态 + 上市日期)
        sb = conn.execute("""
            SELECT ts_code, list_date, name
            FROM stock_basic
        """).fetchdf()

        # 各调仓日市值 (用于排除后 20%)
        dates_sql = "'" + "','".join(rebal_dates) + "'"
        mktcap_df = conn.execute(f"""
            SELECT ts_code, trade_date, total_mv
            FROM daily_basic
            WHERE trade_date IN ({dates_sql})
              AND total_mv > 0
        """).fetchdf()

    sb['trade_date'] = sb['ts_code']  # placeholder
    sb['is_st'] = sb['name'].str.contains('ST', na=False)
    sb['list_date'] = sb['list_date'].apply(_norm_date)
    st_set = set(sb.loc[sb['is_st'], 'ts_code'])
    list_date_map: Dict[str, str] = dict(zip(sb['ts_code'], sb['list_date']))

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
            # 排除 ST
            if ts in st_set:
                continue
            # 排除市值后 20%
            if row['total_mv'] <= cutoff:
                continue
            # 排除上市不满 180 天
            ld = list_date_map.get(ts, '20000101')
            try:
                if (pd.Timestamp(date) - pd.Timestamp(ld)).days < MIN_LISTED_DAYS:
                    continue
            except Exception:
                continue
            elig.add(ts)
        eligible[date] = elig

    avg = np.mean([len(v) for v in eligible.values()])
    print(f"  过滤后平均合规股票数: {avg:.0f} 只/调仓日")
    return eligible


# ── TRANSACTION COSTS ─────────────────────────────────────────────────────────
def trade_cost(value: float, is_sell: bool) -> float:
    """计算单笔交易成本（A 股标准）"""
    v = abs(value)
    c = max(v * COMMISSION_RATE, MIN_COMMISSION)
    t = v * STAMP_TAX_RATE if is_sell else 0.0
    r = v * TRANSFER_RATE
    s = v * SLIPPAGE_RATE
    return c + t + r + s


# ── PORTFOLIO STATE ───────────────────────────────────────────────────────────
class Portfolio:
    """持仓状态管理"""
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.shares:      Dict[str, int]   = {}  # ts_code → 持仓股数
        self.entry_price: Dict[str, float] = {}  # ts_code → 入场均价
        self.entry_date:  Dict[str, str]   = {}  # ts_code → 入场日期
        self.peak_price:  Dict[str, float] = {}  # ts_code → 持仓期最高价
        self.dc_count:    Dict[str, int]   = {}  # ts_code → 连续 MA 死叉天数

    def total_value(self, prices: pd.Series) -> float:
        pos_val = sum(
            self.shares.get(ts, 0) * float(prices.get(ts, 0) or 0)
            for ts in list(self.shares)
        )
        return self.cash + pos_val

    def holding_calendar_days(self, ts_code: str, date: str) -> int:
        ed = self.entry_date.get(ts_code)
        if ed is None:
            return 0
        try:
            return (pd.Timestamp(date) - pd.Timestamp(ed)).days
        except Exception:
            return 0

    def sell(self, ts_code: str, price: float, date: str) -> float:
        """卖出，返回收到的现金（扣成本后）"""
        sh = self.shares.pop(ts_code, 0)
        if sh == 0 or price <= 0:
            return 0.0
        val = sh * price
        cash_in = val - trade_cost(val, is_sell=True)
        self.cash += cash_in
        self.entry_price.pop(ts_code, None)
        self.entry_date.pop(ts_code, None)
        self.peak_price.pop(ts_code, None)
        self.dc_count.pop(ts_code, None)
        return cash_in

    def buy(self, ts_code: str, price: float, target_val: float, date: str) -> bool:
        """买入，target_val 为目标市值；返回是否成功"""
        if price <= 0 or target_val < MIN_TRADE_VALUE:
            return False
        cost_factor = 1 + COMMISSION_RATE + TRANSFER_RATE + SLIPPAGE_RATE
        affordable = min(target_val, self.cash / cost_factor)
        if affordable < MIN_TRADE_VALUE:
            return False
        # 整手 (100 股)，小股 fallback 到 1 股
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
        # 更新持仓（加仓取加权均价）
        old_sh    = self.shares.get(ts_code, 0)
        old_ep    = self.entry_price.get(ts_code, price)
        new_sh    = old_sh + sh
        new_ep    = (old_ep * old_sh + price * sh) / new_sh
        self.shares[ts_code]      = new_sh
        self.entry_price[ts_code] = new_ep
        self.entry_date[ts_code]  = self.entry_date.get(ts_code, date)
        self.peak_price[ts_code]  = max(self.peak_price.get(ts_code, price), price)
        self.dc_count[ts_code]    = self.dc_count.get(ts_code, 0)
        return True


# ── DAILY MONITORING (止损 / 死叉) ────────────────────────────────────────────
def daily_stop_check(portfolio: Portfolio,
                     prices: pd.Series,
                     ma5: pd.Series,
                     ma20: pd.Series,
                     date: str) -> List[str]:
    """
    检查每只持仓的退出条件，返回需卖出的 ts_code 列表。
    不修改持仓状态（仅检测），由调用方执行卖出。
    """
    to_sell = []
    for ts in list(portfolio.shares):
        price = prices.get(ts, np.nan)

        # 停牌/退市 → 强制清仓（使用入场价兜底避免除以0）
        if pd.isna(price) or price <= 0:
            to_sell.append(ts)
            continue

        ep   = portfolio.entry_price.get(ts, price)
        peak = portfolio.peak_price.get(ts, price)

        # 更新追踪最高价
        if price > peak:
            portfolio.peak_price[ts] = price
            peak = price

        # ① 硬止损：距入场价 -8%
        if price < ep * (1 - STOP_LOSS_ENTRY):
            to_sell.append(ts)
            continue

        # ② 追踪止损：距最高点 -7%
        if price < peak * (1 - TRAILING_STOP):
            to_sell.append(ts)
            continue

        # ③ MA 死叉（持有超过 MIN_HOLD_DAYS 后才检查）
        if portfolio.holding_calendar_days(ts, date) >= MIN_HOLD_DAYS:
            m5  = ma5.get(ts, np.nan)
            m20 = ma20.get(ts, np.nan)
            if pd.notna(m5) and pd.notna(m20):
                if m5 < m20:
                    portfolio.dc_count[ts] = portfolio.dc_count.get(ts, 0) + 1
                    if portfolio.dc_count[ts] >= MA_DEATH_DAYS:
                        to_sell.append(ts)
                        continue
                else:
                    portfolio.dc_count[ts] = 0  # 重置计数

    return list(set(to_sell))


# ── REBALANCING ───────────────────────────────────────────────────────────────
def rebalance(portfolio: Portfolio,
              target_stocks: List[str],
              n_slots: int,
              prices: pd.Series,
              date: str) -> int:
    """
    按照 CS 模型打分后的 target_stocks 进行调仓。

    规则:
      n_slots = 0 时 (熊市/MA 规则): 不开新仓，现有持仓由止损自然退出。
      n_slots > 0 时:
        - 目标持仓 = top-N stocks (N = min(n_slots, len(target_stocks)))
        - 不在目标中且持有满 MIN_HOLD_DAYS 的仓位：卖出
        - 目标中尚未持有的股票：均等买入
      权重 = 总净值 / n_slots（固定槽位权重，不随持仓数量变化）
    """
    if n_slots == 0:
        return 0

    total_val    = portfolio.total_value(prices)
    slot_value   = total_val / n_slots
    target_set   = set(target_stocks[:n_slots])
    current_set  = set(portfolio.shares)

    trades = 0

    # ── 卖出：不在目标且持有满期的仓位 ──────────────────────────────────────
    for ts in list(current_set - target_set):
        if portfolio.holding_calendar_days(ts, date) >= MIN_HOLD_DAYS:
            p = prices.get(ts, np.nan)
            if pd.notna(p) and p > 0:
                portfolio.sell(ts, p, date)
                trades += 1

    # ── 买入：目标中尚未持有的股票 ──────────────────────────────────────────
    for ts in list(target_set - current_set):
        p = prices.get(ts, np.nan)
        if pd.isna(p) or p <= 0:
            continue
        if portfolio.buy(ts, p, slot_value, date):
            trades += 1

    return trades


# ── MAIN BACKTEST LOOP ────────────────────────────────────────────────────────
def run_backtest(price_pv:     pd.DataFrame,
                 ma5_pv:       pd.DataFrame,
                 ma20_pv:      pd.DataFrame,
                 vol_pv:       pd.DataFrame,
                 cs_preds:     pd.DataFrame,
                 timing_budget: Dict[str, int],
                 eligible:     Dict[str, set],
                 rebal_set:    set) -> Tuple[pd.DataFrame, List[dict]]:
    """
    日频回测主循环。
    每日: 更新最高价 → 检查止损/死叉 → 记录净值
    调仓日: 计算槽位 → 过滤合规股 → 排除停牌 → 调仓
    """
    print("\n[Backtest] 运行联合策略回测...")
    pf = Portfolio(INITIAL_CAPITAL)

    # 预索引 CS 预测
    cs_by_date: Dict[str, pd.DataFrame] = {
        dt: grp for dt, grp in cs_preds.groupby('trade_date')
    }

    all_dates = sorted(d for d in price_pv.index if d >= BACKTEST_START)
    equity: List[dict] = []
    trade_log: List[dict] = []
    prev_slots = MAX_SLOTS // 2

    for i, date in enumerate(all_dates):
        if i % 50 == 0:
            tv = pf.total_value(price_pv.loc[date])
            print(f"  {date}  净值: ¥{tv:>12,.0f}  持仓: {len(pf.shares):2d} 只")

        prices = price_pv.loc[date]
        ma5    = ma5_pv.loc[date]  if date in ma5_pv.index  else pd.Series(dtype=float)
        ma20   = ma20_pv.loc[date] if date in ma20_pv.index else pd.Series(dtype=float)

        # ── ① 每日止损检查 ──────────────────────────────────────────────────
        exits = daily_stop_check(pf, prices, ma5, ma20, date)
        for ts in exits:
            p = float(prices.get(ts, pf.entry_price.get(ts, 1.0)) or 1.0)
            cash_rcv = pf.sell(ts, p, date)
            trade_log.append({
                'date': date, 'ts_code': ts,
                'action': 'stop_sell', 'price': p, 'cash': cash_rcv
            })

        # ── ② 调仓日 ────────────────────────────────────────────────────────
        if date in rebal_set:
            n_slots = timing_budget.get(date, prev_slots)
            prev_slots = n_slots

            # 合规股票
            elig_set = eligible.get(date, set())

            # CS 预测分数
            scores_df = cs_by_date.get(date, pd.DataFrame(columns=['ts_code', 'pred']))

            # 过滤: 合规 + 未停牌 (vol > 0)
            vol_today = vol_pv.loc[date] if date in vol_pv.index else pd.Series(dtype=float)
            active = set(vol_today[vol_today > 0].dropna().index)
            scores_df = scores_df[
                scores_df['ts_code'].isin(elig_set) &
                scores_df['ts_code'].isin(active)
            ].sort_values('pred', ascending=False)

            target_stocks = scores_df['ts_code'].tolist()

            if n_slots > 0 and target_stocks:
                n_trades = rebalance(pf, target_stocks, n_slots, prices, date)
                trade_log.append({
                    'date': date, 'ts_code': 'REBAL',
                    'action': 'rebalance',
                    'price': n_slots, 'cash': n_trades
                })

        # ── ③ 记录净值 ──────────────────────────────────────────────────────
        tv = pf.total_value(prices)
        equity.append({
            'date': date,
            'total_value': tv,
            'cash': pf.cash,
            'n_positions': len(pf.shares),
        })

    equity_df = pd.DataFrame(equity).set_index('date')
    equity_df.index = pd.to_datetime(equity_df.index, format='%Y%m%d')
    print(f"\n  最终净值: ¥{equity_df['total_value'].iloc[-1]:,.0f}")
    print(f"  平均持仓: {equity_df['n_positions'].mean():.1f} 只")
    print(f"  交易记录: {len(trade_log)} 条")
    return equity_df, trade_log


# ── PERFORMANCE METRICS ───────────────────────────────────────────────────────
def compute_metrics(equity_df: pd.DataFrame) -> dict:
    """计算综合绩效指标"""
    vals = equity_df['total_value']
    rets = vals.pct_change().dropna()
    n    = len(rets)
    tpy  = 244  # A 股年交易日
    yrs  = n / tpy

    total_ret  = vals.iloc[-1] / vals.iloc[0] - 1
    annual_ret = (1 + total_ret) ** (1 / yrs) - 1 if yrs > 0 else 0
    annual_vol = rets.std() * np.sqrt(tpy)
    rf_daily   = 0.02 / tpy
    sharpe     = (rets.mean() - rf_daily) / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0

    peak   = vals.expanding().max()
    dd     = (vals - peak) / peak
    max_dd = dd.min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.nan

    down   = rets[rets < 0]
    sortino = (annual_ret - 0.02) / (down.std() * np.sqrt(tpy)) if len(down) > 0 else np.nan
    win_rate = (rets > 0).mean()

    # 逐年收益
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
        max_dd=max_dd, win_rate=win_rate, years=yrs, annual_rets=annual_rets
    )


def print_metrics(m: dict):
    print("\n" + "═" * 55)
    print("  联合 ML 策略回测报告")
    print("  训练: 2018-2022  |  回测: 2023-2025 (严格样本外)")
    print("═" * 55)
    print(f"  总收益率:     {m['total_ret']:>+.2%}")
    print(f"  年化收益率:   {m['annual_ret']:>+.2%}")
    print(f"  年化波动率:   {m['annual_vol']:>.2%}")
    print(f"  夏普比率:     {m['sharpe']:>.3f}")
    print(f"  索提诺比率:   {m['sortino']:>.3f}")
    print(f"  卡玛比率:     {m['calmar']:>.3f}")
    print(f"  最大回撤:     {m['max_dd']:>.2%}")
    print(f"  日胜率:       {m['win_rate']:>.1%}")
    print(f"  时间跨度:     {m['years']:.2f} 年")
    print("  逐年收益:")
    for yr, ret in m['annual_rets'].items():
        bar = '█' * int(abs(ret) * 100) + (' ↑' if ret > 0 else ' ↓')
        print(f"    {yr}: {ret:>+.2%}  {bar}")
    print("═" * 55)


# ── BENCHMARK (沪深300) ───────────────────────────────────────────────────────
def load_benchmark() -> pd.Series:
    """加载沪深300作为对比基准"""
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT trade_date, close
            FROM index_daily
            WHERE ts_code = '000300.SH'
              AND trade_date >= '{BACKTEST_START}'
              AND trade_date <= '{BACKTEST_END}'
            ORDER BY trade_date
        """).fetchdf()
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    s = df.set_index('trade_date')['close']
    s.index = pd.to_datetime(s.index, format='%Y%m%d')
    return s / s.iloc[0]  # 归一化为 NAV


# ── PLOTTING ──────────────────────────────────────────────────────────────────
def plot_results(equity_df: pd.DataFrame, m: dict, benchmark: pd.Series):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor='#f8f8f8')

    nav = equity_df['total_value'] / INITIAL_CAPITAL

    # ── ① 净值曲线 ───────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(nav.index, nav.values, color='#1a6db4', lw=1.8, label='联合ML策略')
    # 对齐基准到相同日期
    bm_aligned = benchmark.reindex(nav.index, method='ffill').dropna()
    if len(bm_aligned):
        ax1.plot(bm_aligned.index, bm_aligned.values,
                 color='#e05c2a', lw=1.2, alpha=0.7, ls='--', label='沪深300')
    ax1.axhline(1.0, color='gray', ls=':', alpha=0.4)
    ax1.set_title(f"联合ML策略 NAV  |  年化 {m['annual_ret']:+.2%}  "
                  f"最大回撤 {m['max_dd']:.2%}  夏普 {m['sharpe']:.3f}",
                  fontsize=12, pad=8)
    ax1.set_ylabel('NAV (初始=1.0)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ── ② 回撤曲线 ──────────────────────────────────────────────────────────
    ax2 = axes[1]
    peak = equity_df['total_value'].expanding().max()
    dd   = (equity_df['total_value'] - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color='#c0392b', alpha=0.35)
    ax2.plot(dd.index, dd.values, color='#c0392b', lw=0.8)
    ax2.axhline(0, color='gray', ls=':', alpha=0.4)
    ax2.set_ylabel('回撤 (%)')
    ax2.set_title(f"最大回撤 {m['max_dd']:.2%}", fontsize=10)
    ax2.grid(True, alpha=0.25)

    # ── ③ 持仓数量 ──────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(equity_df.index, equity_df['n_positions'].values,
                     color='#27ae60', alpha=0.4)
    ax3.plot(equity_df.index, equity_df['n_positions'].values,
             color='#27ae60', lw=0.8)
    ax3.set_ylabel('持仓只数')
    ax3.set_xlabel('日期')
    ax3.set_title(f"持仓数量（均值 {equity_df['n_positions'].mean():.1f} 只）", fontsize=10)
    ax3.grid(True, alpha=0.25)

    # 年度分隔线
    for ax in axes:
        for yr in range(2023, 2027):
            ax.axvline(pd.Timestamp(f'{yr}-01-01'), color='gray', ls=':', alpha=0.3)

    plt.tight_layout(pad=2.0)
    out_path = OUT_DIR / 'combined_strategy_equity.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表已保存: {out_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  联合 ML 策略回测")
    print("  截面选股 + 时序择时 + 规则兜底")
    print("  训练: 2018-2022  |  回测: 2023-2025 (严格 OOS)")
    print("=" * 60)

    # 1. 加载预测结果 (均为严格样本外)
    cs_preds    = load_cs_predictions()
    timing_agg  = load_timing_predictions()

    # 2. 上证指数 + 市场状态
    sse_df = load_sse_index()

    # 3. 日线价格数据
    price_pv, ma5_pv, ma20_pv, vol_pv = load_price_data()

    # 4. 调仓日期 = CS 预测中的调仓日
    all_rebal = sorted(cs_preds['trade_date'].unique())
    rebal_dates = [d for d in all_rebal
                   if BACKTEST_START <= d <= BACKTEST_END]
    rebal_set = set(rebal_dates)
    print(f"\n调仓日期: {len(rebal_dates)} 个 ({rebal_dates[0]} ~ {rebal_dates[-1]})")

    # 5. 计算每个调仓日的仓位预算 (时序模型 + MA 规则)
    print("\n计算仓位预算...")
    timing_budget = compute_position_budget(timing_agg, sse_df, rebal_dates)

    # 6. 安全过滤器 (ST / 停牌 / 新股 / 小市值)
    print("\n计算安全过滤器...")
    eligible = load_safety_filters(rebal_dates)

    # 7. 回测
    equity_df, trade_log = run_backtest(
        price_pv, ma5_pv, ma20_pv, vol_pv,
        cs_preds, timing_budget, eligible, rebal_set
    )

    # 8. 绩效指标
    m = compute_metrics(equity_df)
    print_metrics(m)

    # 9. 加载基准
    try:
        benchmark = load_benchmark()
    except Exception:
        benchmark = pd.Series(dtype=float)

    # 10. 绘图
    plot_results(equity_df, m, benchmark)

    # 11. 保存结果
    eq_path = OUT_DIR / 'combined_equity.csv'
    equity_df.to_csv(eq_path)
    print(f"  净值数据已保存: {eq_path}")

    # 保存指标摘要
    summary_rows = {
        '总收益率': f"{m['total_ret']:+.4%}",
        '年化收益率': f"{m['annual_ret']:+.4%}",
        '年化波动率': f"{m['annual_vol']:.4%}",
        '夏普比率': f"{m['sharpe']:.4f}",
        '索提诺比率': f"{m['sortino']:.4f}",
        '卡玛比率': f"{m['calmar']:.4f}",
        '最大回撤': f"{m['max_dd']:.4%}",
        '日胜率': f"{m['win_rate']:.4%}",
    }
    for yr, ret in m['annual_rets'].items():
        summary_rows[f'{yr}年收益'] = f"{ret:+.4%}"
    summary_df = pd.DataFrame.from_dict(summary_rows, orient='index', columns=['值'])
    summary_path = OUT_DIR / 'combined_metrics.csv'
    summary_df.to_csv(summary_path)
    print(f"  指标摘要已保存: {summary_path}")

    print("\n✓ 完成！所有结果保存至 output/combined/")


if __name__ == '__main__':
    main()
