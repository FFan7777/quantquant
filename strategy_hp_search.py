#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略级超参数搜索  Strategy Hyperparameter Search  v2
=======================================================

核心改进（基于 Regime Overfitting 实验教训）：
  1. 多周期 Val（2022-2024）：跨越熊/震荡/牛三种 Regime，不再过拟合单年
  2. Minimax 目标函数：最大化最差年份 Calmar，强制参数对所有 Regime 稳健
  3. Regime-Switching 退出参数：bull/neutral/bear 三档动态 MA 退出参数
  4. 精简搜索空间：mktcap_pct_cut 实验证明无效，已从搜索中移除
  5. 年度惩罚机制：某年出现大亏损直接返回极低分数

数据划分（与 xgboost_cross_section.py LONG_VAL=True 一致）：
  Train end: 2021-12-31   （XGBoost 训练，不参与策略 HP 搜索）
  Val:       2022-02-01 ~ 2024-12-31  ← 策略 HP 搜索（3年，多 Regime）
  Test:      2025-02-01 ~ 2026-03-16  ← 唯一真实 OOS 评测

搜索参数（坐标下降）：
  Step A: (max_slots, half_slots) — 持仓容量
  Step B: stop_loss               — 止损幅度
  Step C: (bear_ma_death, bear_min_hold)    — 熊市退出（最敏感）
  Step D: (bull_ma_death, bull_min_hold)    — 牛市退出
  Step E: (neutral_ma_death, neutral_min_hold) — 中性退出

数据依赖：
  output/csv/xgb_cs_pred_val.csv   ← LONG_VAL=True 下由 xgboost_cross_section.py 生成
                                      覆盖 2022-2024（train-only 模型，严格 PIT）
  output/csv/xgb_cross_section_predictions.csv  ← 同上（train+val 模型，覆盖 2025+）
  output/csv/index_timing_predictions.csv       ← index_timing_model.py 生成
"""

import json
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
# 1. 全局路径与日期常量
# ══════════════════════════════════════════════════════════════════════
DB_PATH = config.db_path

# Val：多周期（2022-2024），跨越熊/震荡/牛三种 Regime
VAL_START  = "20220201"   # 含20日隔离（train end 2021-12-31 + embargo）
VAL_END    = "20241231"
VAL_YEARS  = [2022, 2023, 2024]   # 用于 minimax 目标函数

# Test：唯一 OOS
TEST_START = "20250201"   # 含20日隔离（val end 2024-12-31 + embargo）
TEST_END   = "20260316"

PRICE_WARMUP = "20211101"   # 为 2022 的 MA 计算提供预热数据

INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE  = 0.001
TRANSFER_RATE   = 0.00002
SLIPPAGE_RATE   = 0.0001
MIN_COMMISSION  = 5.0
MIN_TRADE_VALUE = 2000.0
RISK_FREE_RATE  = 0.02
MKTCAP_PCT_CUT  = 10   # 固定，实验证明搜索无收益

OUT_DIR = ROOT / "output" / "strategy_hp_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 2. 策略参数 dataclass（含 Regime-Switching 退出参数）
# ══════════════════════════════════════════════════════════════════════
@dataclass
class StrategyParams:
    # 持仓容量
    max_slots:  int   = 8     # 满仓（slots=20）最多持股数
    half_slots: int   = 3     # 半仓（slots=10）最多持股数
    # 止损
    stop_loss:  float = 0.08  # 入场价回撤 X% → 硬止损（所有 Regime 统一）
    trailing_stop: float = 1.00  # 追踪止损（1.0=禁用）
    # Regime-Switching 退出参数
    bull_ma_death:    int = 7   # slots=20：MA5<MA20 连续N天退出（宽松）
    bull_min_hold:    int = 7   # slots=20：最短持有期
    neutral_ma_death: int = 5   # slots=10：中性
    neutral_min_hold: int = 5
    bear_ma_death:    int = 3   # slots=0：快速退出（严格）
    bear_min_hold:    int = 3
    # 其他
    slot_confirm_days: int  = 3     # 连续N天slots>0才开新仓（防whipsaw）
    use_vol_scale:     bool = True  # 风险平价加权
    use_signal_scale:  bool = True  # 信号强度加权

    def label(self) -> str:
        return (f"ms{self.max_slots}hs{self.half_slots}_"
                f"sl{int(self.stop_loss*100)}_"
                f"bull({self.bull_ma_death},{self.bull_min_hold})_"
                f"bear({self.bear_ma_death},{self.bear_min_hold})")


DEFAULT_PARAMS = StrategyParams()


# ══════════════════════════════════════════════════════════════════════
# 3. 数据加载（一次性，供所有回测复用）
# ══════════════════════════════════════════════════════════════════════
def _norm_date(s) -> str:
    return str(s).replace('-', '')


def load_price_data(end_date: str) -> Tuple:
    """返回 (price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv)"""
    print("  加载价格数据（含2021年预热期）...")
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
    pct_chg_pv  = price_pv.pct_change()
    print(f"    价格矩阵: {len(price_pv)} 天 × {len(price_pv.columns):,} 只")
    return price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv


def load_safety_info(rebal_dates: List[str]) -> Dict[str, set]:
    """预加载所有调仓日的合规股票集合（固定 mktcap_pct_cut=10%，不再搜索）"""
    print(f"  加载合规股信息（mktcap_pct_cut={MKTCAP_PCT_CUT}%，固定）...")
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
        cutoff = np.percentile(sub['total_mv'].values, MKTCAP_PCT_CUT)
        elig = set()
        for _, row in sub.iterrows():
            ts = row['ts_code']
            if ts in st_set or row['total_mv'] <= cutoff:
                continue
            ld = list_date_map.get(ts, '20000101')
            try:
                if (pd.Timestamp(date) - pd.Timestamp(ld)).days < 90:
                    continue
            except Exception:
                continue
            elig.add(ts)
        eligible[date] = elig
    avg = np.mean([len(v) for v in eligible.values()]) if eligible else 0
    print(f"    平均合规股票数: {avg:.0f} 只/调仓日")
    return eligible


def load_cs_predictions(period: str = "val") -> pd.DataFrame:
    """
    加载截面选股预测，并在 H5 预测文件存在时做 H10+H5 ensemble（0.7×H10 rank + 0.3×H5 rank）。
    period='val'(2022-2024) 或 'test'(2025+)
    """
    if period == "val":
        fpath_h10  = ROOT / "output" / "csv" / "xgb_cs_pred_val.csv"
        fpath_h5   = ROOT / "output" / "csv" / "xgb_cs_pred_val_h5.csv"
        start, end = VAL_START, VAL_END
        label = "Val 2022-2024"
    else:
        fpath_h10  = ROOT / "output" / "csv" / "xgb_cross_section_predictions.csv"
        fpath_h5   = ROOT / "output" / "csv" / "xgb_cs_pred_h5.csv"
        start, end = TEST_START, TEST_END
        label = "Test 2025+"

    if not fpath_h10.exists():
        raise FileNotFoundError(
            f"找不到预测文件: {fpath_h10}\n"
            f"请先运行 xgboost_cross_section.py（LONG_VAL=True）"
        )
    df = pd.read_csv(fpath_h10, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    df = df[(df['trade_date'] >= start) & (df['trade_date'] <= end)].copy()

    # H10+H5 ensemble（与 index_ma_combined_strategy.py 一致：0.7×H10 + 0.3×H5）
    if fpath_h5.exists():
        df5 = pd.read_csv(fpath_h5, dtype={'trade_date': str})
        df5['trade_date'] = df5['trade_date'].apply(_norm_date)
        df5 = df5[(df5['trade_date'] >= start) & (df5['trade_date'] <= end)]
        df5 = df5.rename(columns={'pred': 'pred_h5'})
        df = df.merge(df5[['ts_code', 'trade_date', 'pred_h5']],
                      on=['ts_code', 'trade_date'], how='left')
        df['r10'] = df.groupby('trade_date')['pred'].rank(pct=True)
        df['r5']  = df.groupby('trade_date')['pred_h5'].rank(pct=True)
        df['pred'] = 0.7 * df['r10'] + 0.3 * df['r5'].fillna(df['r10'])
        print(f"  CS预测({label}) H10+H5 ensemble: {len(df):,}行  "
              f"{df['trade_date'].nunique()}个截面  {df['ts_code'].nunique()}只股票")
    else:
        print(f"  CS预测({label}) H10 only（H5文件不存在）: {len(df):,}行  "
              f"{df['trade_date'].nunique()}个截面  {df['ts_code'].nunique()}只股票")

    return df[['ts_code', 'trade_date', 'pred']].copy()


def load_index_timing(period: str = "val") -> pd.Series:
    """加载指数择时信号 slots∈{0,10,20}"""
    fpath = ROOT / "output" / "csv" / "index_timing_predictions.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"找不到: {fpath}")
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
# 4. 参数化回测引擎（含 Regime-Switching）
# ══════════════════════════════════════════════════════════════════════
def trade_cost(value: float, is_sell: bool) -> float:
    v = abs(value)
    c = max(v * COMMISSION_RATE, MIN_COMMISSION)
    t = v * STAMP_TAX_RATE if is_sell else 0.0
    return c + t + v * TRANSFER_RATE + v * SLIPPAGE_RATE


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
        val     = sh * price
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
        sh    = int(affordable / price / 100) * 100
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
                     p: StrategyParams, slots_today: int) -> List[str]:
    """
    Regime-Switching 风控：根据 slots_today 选择 MA 退出参数
      slots=20 → bull params（宽松）
      slots=10 → neutral params（均衡）
      slots=0  → bear params（快速退出）
    """
    if slots_today == 20:
        ma_death = p.bull_ma_death
        min_hold = p.bull_min_hold
    elif slots_today == 10:
        ma_death = p.neutral_ma_death
        min_hold = p.neutral_min_hold
    else:
        ma_death = p.bear_ma_death
        min_hold = p.bear_min_hold

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
        # 硬止损（所有 Regime 统一）
        if price < ep * (1 - p.stop_loss):
            to_sell.append(ts)
            continue
        # 追踪止损（默认禁用）
        if price < peak * (1 - p.trailing_stop):
            to_sell.append(ts)
            continue
        # MA 死叉（Regime-Switching）
        hd = portfolio.hold_days.get(ts, 0)
        if hd >= min_hold:
            m5  = ma5.get(ts, np.nan)
            m20 = ma20.get(ts, np.nan)
            if pd.notna(m5) and pd.notna(m20):
                if m5 < m20:
                    portfolio.dc_count[ts] = portfolio.dc_count.get(ts, 0) + 1
                    if portfolio.dc_count[ts] >= ma_death:
                        to_sell.append(ts)
                        continue
                else:
                    portfolio.dc_count[ts] = 0
    return list(set(to_sell))


def rebalance(portfolio: Portfolio, target_stocks: List[str],
              n_slots: int, prices: pd.Series, date: str,
              slot_confirmed: bool, ret_vol: Optional[pd.Series],
              p: StrategyParams) -> int:
    """按 CS 打分调仓。n_slots=0 不开仓。"""
    if n_slots == 0:
        return 0
    total_val  = portfolio.total_value(prices)
    target_set = set(target_stocks[:n_slots])
    trades     = 0

    # 卖出不在目标中的持仓（已过 min_hold 的 neutral 设定，用于调仓日强制换股）
    neutral_min = p.neutral_min_hold
    for ts in sorted(set(portfolio.shares) - target_set):
        if portfolio.hold_days.get(ts, 0) >= neutral_min:
            price = prices.get(ts, np.nan)
            if pd.notna(price) and price > 0:
                portfolio.sell(ts, price, date)
                trades += 1

    if slot_confirmed:
        target_list = target_stocks[:n_slots]
        top_stocks  = [ts for ts in target_list if ts not in portfolio.shares]
        fixed_sv    = total_val / p.max_slots

        # 信号强度加权 [1.5x → 0.5x]
        n = len(target_list)
        if p.use_signal_scale and n > 0:
            sig_map = {ts: (1.5 - 1.0 * i / max(n - 1, 1))
                       for i, ts in enumerate(target_list)}
        else:
            sig_map = {ts: 1.0 for ts in target_list}

        # 风险平价加权
        if p.use_vol_scale and ret_vol is not None and top_stocks:
            vols = np.array([float(ret_vol.get(ts, np.nan) or np.nan)
                             for ts in top_stocks])
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
) -> pd.DataFrame:
    """日频回测主循环（完全参数化，含 Regime-Switching 退出）"""
    pf         = Portfolio(INITIAL_CAPITAL)
    cs_by_date = {dt: grp for dt, grp in cs_preds.groupby('trade_date')}
    rebal_set  = set(cs_preds['trade_date'].unique())
    all_dates  = sorted(d for d in price_pv.index if start_date <= d <= end_date)

    equity         = []
    consec_nonzero = 0

    for date in all_dates:
        slots_today    = _get_slots(index_slots, date, p.neutral_min_hold)
        if slots_today > 0:
            consec_nonzero += 1
        else:
            consec_nonzero = 0

        prices = price_pv.loc[date]
        ma5    = ma5_pv.loc[date]  if date in ma5_pv.index  else pd.Series(dtype=float)
        ma20   = ma20_pv.loc[date] if date in ma20_pv.index else pd.Series(dtype=float)

        # 每日风控（Regime-Switching）
        exits = daily_stop_check(pf, prices, ma5, ma20, p, slots_today)
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
                limit_up = set(
                    pct_chg_pv.loc[date][pct_chg_pv.loc[date] >= 0.095].dropna().index)
                scores_df = scores_df[~scores_df['ts_code'].isin(limit_up)]

            target_stocks  = scores_df['ts_code'].tolist()
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
# 5. 绩效指标 & Minimax 目标函数
# ══════════════════════════════════════════════════════════════════════
def _year_calmar(equity_df: pd.DataFrame, year: int) -> Optional[float]:
    """计算单年 Calmar 比率。数据不足返回 None。"""
    mask = equity_df.index.year == year
    if mask.sum() < 20:
        return None
    vals = equity_df.loc[mask, 'total_value']
    ret  = vals.iloc[-1] / vals.iloc[0] - 1
    peak = vals.expanding().max()
    dd   = float(((vals - peak) / peak).min())
    if dd == 0:
        return 10.0 if ret > 0 else 0.0
    return ret / abs(dd)


def compute_metrics(equity_df: pd.DataFrame) -> dict:
    """全期绩效 + 逐年数据"""
    vals = equity_df['total_value']
    rets = vals.pct_change().dropna()
    n    = len(rets)
    tpy  = 244
    yrs  = n / tpy
    if yrs <= 0:
        return dict(annual_ret=0, max_dd=0, sharpe=0, calmar=0,
                    total_ret=0, years=0, annual_vol=0, annual_rets={},
                    yearly_calmars={})

    total_ret  = vals.iloc[-1] / vals.iloc[0] - 1
    annual_ret = (1 + total_ret) ** (1 / yrs) - 1
    annual_vol = rets.std() * np.sqrt(tpy)
    rf_daily   = 0.02 / tpy
    sharpe     = (rets.mean() - rf_daily) / rets.std() * np.sqrt(tpy) if rets.std() > 0 else 0
    peak       = vals.expanding().max()
    dd         = (vals - peak) / peak
    max_dd     = float(dd.min())
    calmar     = annual_ret / abs(max_dd) if max_dd != 0 else np.nan

    annual_rets = {}
    yearly_calmars = {}
    for yr in range(2022, 2027):
        mask = equity_df.index.year == yr
        if mask.sum() < 5:
            continue
        yv = vals[mask]
        annual_rets[str(yr)] = yv.iloc[-1] / yv.iloc[0] - 1
        yc = _year_calmar(equity_df, yr)
        if yc is not None:
            yearly_calmars[str(yr)] = yc

    return dict(total_ret=total_ret, annual_ret=annual_ret, annual_vol=annual_vol,
                sharpe=sharpe, calmar=calmar, max_dd=max_dd, years=yrs,
                annual_rets=annual_rets, yearly_calmars=yearly_calmars)


def minimax_score(equity_df: pd.DataFrame) -> float:
    """
    Minimax 目标：最大化 Val 期内最差年份的 Calmar 比率。
    - 任一年亏损超过 10% → 重惩罚（返回 -10）
    - 任一年亏损 > 0     → 轻惩罚（Calmar 扣分）
    - 全年正收益          → 返回 min(yearly_calmar)
    """
    calmars = {}
    for yr in VAL_YEARS:
        c = _year_calmar(equity_df, yr)
        if c is None:
            continue
        calmars[yr] = c

    if not calmars:
        return -999.0

    # 获取各年收益
    vals = equity_df['total_value']
    yearly_rets = {}
    for yr in VAL_YEARS:
        mask = equity_df.index.year == yr
        if mask.sum() < 5:
            continue
        yv = vals[mask]
        yearly_rets[yr] = yv.iloc[-1] / yv.iloc[0] - 1

    # 惩罚：某年大亏
    for yr, ret in yearly_rets.items():
        if ret < -0.10:   # 亏超10%，重惩罚
            return min(calmars.values()) - 10.0
        if ret < 0:       # 亏损但<10%，轻惩罚
            return min(calmars.values()) - 2.0

    return float(min(calmars.values()))


# ══════════════════════════════════════════════════════════════════════
# 6. 坐标下降搜索
# ══════════════════════════════════════════════════════════════════════
def run_one(label: str, p: StrategyParams,
            cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
            vol_pv, retvol20_pv, pct_chg_pv, eligible,
            start_date: str, end_date: str) -> dict:
    eq = run_backtest(p, cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
                      vol_pv, retvol20_pv, pct_chg_pv, eligible,
                      start_date, end_date)
    m = compute_metrics(eq)
    m['label']     = label
    m['params']    = asdict(p)
    m['equity_df'] = eq
    m['minimax']   = minimax_score(eq)
    return m


def search_dimension(name: str, dim_overrides: list,
                     base: StrategyParams,
                     cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
                     vol_pv, retvol20_pv, pct_chg_pv, eligible,
                     start_date: str, end_date: str,
                     counter: list) -> Tuple[StrategyParams, list]:
    """
    坐标下降单维度搜索。dim_overrides 是 dict 列表，每个只指定被搜索的字段。
    其余字段继承 base（保留前序步骤的最优值）。
    目标：最大化 minimax_score（最差年份 Calmar）。
    """
    print(f"\n── 搜索维度: {name}  base={base.label()} ──")
    results = []
    for override in dim_overrides:
        p = StrategyParams(**{**asdict(base), **override})
        m = run_one(p.label(), p, cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
                    vol_pv, retvol20_pv, pct_chg_pv, eligible, start_date, end_date)
        results.append(m)
        counter[0] += 1
        # 逐年 Calmar
        yc = m['yearly_calmars']
        yc_str = "  ".join(f"{yr}={v:+.2f}" for yr, v in sorted(yc.items()))
        print(f"  [{counter[0]:2d}] {p.label():<55s}  "
              f"minimax={m['minimax']:>+.2f}  overall={m['calmar']:>+.2f}  "
              f"[{yc_str}]")

    best = max(results, key=lambda x: x['minimax'])
    print(f"  → 最优: {best['label']}  minimax={best['minimax']:.3f}")
    return StrategyParams(**best['params']), results


# ══════════════════════════════════════════════════════════════════════
# 7. 敏感性测试
# ══════════════════════════════════════════════════════════════════════
def sensitivity_test(best_p: StrategyParams,
                     cs_preds, index_slots, price_pv, ma5_pv, ma20_pv,
                     vol_pv, retvol20_pv, pct_chg_pv, eligible,
                     start_date: str, end_date: str) -> pd.DataFrame:
    """对每个核心参数做±1步扰动，验证稳健性（minimax score 变化 > 0.5 视为不稳定）"""
    INSTABILITY = 0.5
    print(f"\n[敏感性测试] 逐维度扰动最优参数（minimax Δ > {INSTABILITY} 视为不稳定）")

    base_m = run_one("base", best_p, cs_preds, index_slots,
                     price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
                     pct_chg_pv, eligible, start_date, end_date)
    base_score = base_m['minimax']
    print(f"  基准 minimax={base_score:.3f}  params={best_p}")

    dims = {
        "max_slots":        [4, 6, 8, 10, 12, 15],
        "half_slots":       [2, 3, 4, 5, 6],
        "stop_loss":        [0.05, 0.06, 0.07, 0.08, 0.10, 0.12],
        "bull_ma_death":    [4, 5, 6, 7, 9, 12],
        "bull_min_hold":    [4, 5, 6, 7, 9, 12],
        "bear_ma_death":    [2, 3, 4, 5],
        "bear_min_hold":    [2, 3, 4, 5],
        "neutral_ma_death": [3, 4, 5, 6, 7],
        "neutral_min_hold": [3, 4, 5, 6, 7],
    }

    rows = []
    for dim, candidates in dims.items():
        best_val = getattr(best_p, dim)
        for v in candidates:
            test_p = StrategyParams(**{**asdict(best_p), dim: v})
            if test_p.half_slots >= test_p.max_slots:
                continue
            m      = run_one(f"{dim}={v}", test_p, cs_preds, index_slots,
                             price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
                             pct_chg_pv, eligible, start_date, end_date)
            delta  = m['minimax'] - base_score
            stable = abs(delta) <= INSTABILITY
            flag   = "✓" if stable else "⚠"
            best_mark = "[best]" if v == best_val else "      "
            print(f"  {dim}={str(v):<6} {best_mark}  minimax={m['minimax']:>+.3f}  "
                  f"Δ={delta:>+.3f}  {flag}")
            rows.append({'dim': dim, 'value': v, 'is_best': (v == best_val),
                         'minimax': m['minimax'], 'delta': delta,
                         'annual_ret': m['annual_ret'], 'max_dd': m['max_dd'],
                         'stable': stable})

        dim_rows = [r for r in rows if r['dim'] == dim]
        vals_    = [r['minimax'] for r in dim_rows if np.isfinite(r['minimax'])]
        rng      = max(vals_) - min(vals_) if vals_ else 0
        stab     = "✓ 稳健" if rng <= INSTABILITY else "⚠ 敏感"
        print(f"  → {dim}: range={rng:.3f}  {stab}\n")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 8. 可视化
# ══════════════════════════════════════════════════════════════════════
def plot_equity(val_eq: pd.DataFrame, test_eq: pd.DataFrame,
                csi300_val: pd.Series, csi300_test: pd.Series,
                best_p: StrategyParams):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Regime-Switching Strategy HP Search\nbest: {best_p.label()}", fontsize=10)

    for ax, eq, csi, period in [
        (axes[0], val_eq, csi300_val, "Val 2022-2024（HP选择集）"),
        (axes[1], test_eq, csi300_test, "Test 2025+（真实OOS）"),
    ]:
        nav = eq['total_value'] / eq['total_value'].iloc[0]
        ax.plot(nav.index, nav.values, label="Strategy", color="steelblue", linewidth=1.5)
        if len(csi) > 0:
            csi_norm = csi / csi.iloc[0]
            ax.plot(pd.to_datetime(csi_norm.index, format='%Y%m%d'),
                    csi_norm.values, label="CSI300", color="gray", linewidth=1, alpha=0.7)
        ax.axhline(1, color='black', linewidth=0.5)
        m = compute_metrics(eq)
        # 标注各年区间
        for yr in range(2022, 2027):
            mask = eq.index.year == yr
            if mask.sum() < 5:
                continue
            xpos = eq.index[mask][len(eq.index[mask])//2]
            yval = float(nav[mask].mean())
            ret  = m['annual_rets'].get(str(yr), np.nan)
            if np.isfinite(ret):
                ax.annotate(f"{yr}\n{ret:+.0%}", xy=(xpos, yval * 0.95),
                            fontsize=7, ha='center', color='darkblue')
        ax.set_title(f"{period}\nann={m['annual_ret']:+.1%}  DD={m['max_dd']:.1%}  "
                     f"Calmar={m['calmar']:.2f}  minimax={minimax_score(eq):.2f}",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylabel("归一化净值")

    plt.tight_layout()
    out = OUT_DIR / "strategy_hp_equity_v2.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  净值图已保存: {out}")


def plot_yearly_calmar(val_results: list):
    """柱状图：所有候选参数的逐年 Calmar，直观展示 minimax 效果"""
    fig, axes = plt.subplots(1, len(VAL_YEARS), figsize=(5 * len(VAL_YEARS), 5))
    fig.suptitle("逐年 Calmar vs 参数组合（Minimax 搜索）", fontsize=11)

    labels = [r['label'][:40] for r in val_results]
    minimax_vals = [r['minimax'] for r in val_results]
    best_idx = int(np.argmax(minimax_vals))

    for ax, yr in zip(axes, VAL_YEARS):
        vals = [r['yearly_calmars'].get(str(yr), np.nan) for r in val_results]
        colors = ['#d62728' if v < 0 else
                  ('#2ca02c' if i == best_idx else '#1f77b4')
                  for i, v in enumerate(vals)]
        ax.barh(range(len(labels)), vals, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f"{yr}年 Calmar", fontsize=9)
        ax.set_xlabel("Calmar")

    plt.tight_layout()
    out = OUT_DIR / "strategy_hp_yearly_calmar.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  逐年Calmar图已保存: {out}")


# ══════════════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════════════
def main():
    import time
    t0 = time.time()
    print("=" * 70)
    print("  策略超参数搜索 v2")
    print("  Val: 2022-2024（多Regime）  目标: Minimax(年度Calmar)")
    print("  Test: 2025+（唯一OOS）")
    print("=" * 70)

    # ── 数据加载（一次性）──────────────────────────────────────────────
    print("\n[数据加载]")
    price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, pct_chg_pv = \
        load_price_data(TEST_END)
    cs_val  = load_cs_predictions("val")
    cs_test = load_cs_predictions("test")
    slot_val  = load_index_timing("val")
    slot_test = load_index_timing("test")
    csi300_val  = load_csi300("val")
    csi300_test = load_csi300("test")

    all_rebal_dates = sorted(set(cs_val['trade_date'].unique()) |
                             set(cs_test['trade_date'].unique()))
    eligible = load_safety_info(all_rebal_dates)

    counter = [0]

    # ── Step A: (max_slots, half_slots) ───────────────────────────────
    slot_overrides = [
        {'max_slots': ms, 'half_slots': hs}
        for ms, hs in [(4,2), (6,2), (6,3), (8,2), (8,3), (8,4), (10,3), (10,4), (10,5),
                       (12,4), (12,5), (15,5), (15,6)]
    ]
    best_p, res_A = search_dimension(
        "max_slots × half_slots", slot_overrides, DEFAULT_PARAMS,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
        pct_chg_pv, eligible, VAL_START, VAL_END, counter)

    # ── Step B: stop_loss ─────────────────────────────────────────────
    stop_overrides = [{'stop_loss': s} for s in [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]]
    best_p, res_B = search_dimension(
        "stop_loss", stop_overrides, best_p,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
        pct_chg_pv, eligible, VAL_START, VAL_END, counter)

    # ── Step C: bear Regime 退出参数（最关键）─────────────────────────
    bear_overrides = [
        {'bear_ma_death': md, 'bear_min_hold': mh}
        for md, mh in [(2,2), (2,3), (3,2), (3,3), (3,5), (4,3), (5,3), (5,5)]
    ]
    best_p, res_C = search_dimension(
        "bear (ma_death, min_hold)", bear_overrides, best_p,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
        pct_chg_pv, eligible, VAL_START, VAL_END, counter)

    # ── Step D: bull Regime 退出参数（让利润奔跑）────────────────────
    bull_overrides = [
        {'bull_ma_death': md, 'bull_min_hold': mh}
        for md, mh in [(4,5), (5,5), (5,7), (7,5), (7,7), (7,10), (10,7), (10,10)]
    ]
    best_p, res_D = search_dimension(
        "bull (ma_death, min_hold)", bull_overrides, best_p,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
        pct_chg_pv, eligible, VAL_START, VAL_END, counter)

    # ── Step E: neutral Regime 退出参数 ──────────────────────────────
    neutral_overrides = [
        {'neutral_ma_death': md, 'neutral_min_hold': mh}
        for md, mh in [(3,3), (3,5), (4,4), (5,3), (5,5), (5,7), (7,5)]
    ]
    best_p, res_E = search_dimension(
        "neutral (ma_death, min_hold)", neutral_overrides, best_p,
        cs_val, slot_val, price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
        pct_chg_pv, eligible, VAL_START, VAL_END, counter)

    # ── 汇总搜索结果 ──────────────────────────────────────────────────
    all_results = res_A + res_B + res_C + res_D + res_E
    results_df  = pd.DataFrame([
        {k: v for k, v in m.items()
         if k not in ('params', 'equity_df', 'annual_rets', 'yearly_calmars')}
        for m in all_results
    ])
    results_df.to_csv(OUT_DIR / "hp_search_results_v2.csv", index=False)

    print(f"\n\n{'='*70}")
    print(f"  最优策略超参（Val 2022-2024，Minimax）: {best_p}")
    print(f"{'='*70}")

    # ── 敏感性测试 ────────────────────────────────────────────────────
    print("\n[敏感性测试] 在 Val (2022-2024) 上逐维度扰动")
    sens_df = sensitivity_test(
        best_p, cs_val, slot_val, price_pv, ma5_pv, ma20_pv,
        vol_pv, retvol20_pv, pct_chg_pv, eligible, VAL_START, VAL_END)
    sens_df.to_csv(OUT_DIR / "sensitivity_results_v2.csv", index=False)

    # ── Val 最终评估 ──────────────────────────────────────────────────
    print("\n[Val 最终评估]")
    val_m = run_one("val_best", best_p, cs_val, slot_val, price_pv, ma5_pv, ma20_pv,
                    vol_pv, retvol20_pv, pct_chg_pv, eligible, VAL_START, VAL_END)
    val_eq = val_m['equity_df']

    # ── Test 评估（唯一真实 OOS）──────────────────────────────────────
    print("\n[Test OOS 评估] 使用 Val 搜出的最优参数，严格 OOS")
    test_m = run_one("test_best", best_p, cs_test, slot_test, price_pv, ma5_pv, ma20_pv,
                     vol_pv, retvol20_pv, pct_chg_pv, eligible, TEST_START, TEST_END)
    test_eq = test_m['equity_df']

    # ── 对比：默认参数在 Test 上 ──────────────────────────────────────
    print("\n[参考] 默认参数在 Test 上的表现:")
    default_m = run_one("test_default", DEFAULT_PARAMS, cs_test, slot_test,
                        price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv,
                        pct_chg_pv, eligible, TEST_START, TEST_END)

    # ── 打印报告 ──────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  策略超参搜索 v2 报告")
    print("═" * 70)
    print(f"\n  最优参数:")
    print(f"    max_slots={best_p.max_slots}  half_slots={best_p.half_slots}")
    print(f"    stop_loss={best_p.stop_loss:.0%}")
    print(f"    bull  regime: ma_death={best_p.bull_ma_death}  min_hold={best_p.bull_min_hold}")
    print(f"    neutral:      ma_death={best_p.neutral_ma_death}  min_hold={best_p.neutral_min_hold}")
    print(f"    bear  regime: ma_death={best_p.bear_ma_death}  min_hold={best_p.bear_min_hold}")

    for label, m in [("Val 2022-2024 [HP选择集]", val_m),
                     ("Test 2025+  [真实OOS]",    test_m),
                     ("Test 默认参数 [参考]",       default_m)]:
        print(f"\n  ── {label} ──")
        print(f"    总收益:  {m['total_ret']:+.2%}")
        print(f"    年化:    {m['annual_ret']:+.2%}")
        print(f"    最大DD:  {m['max_dd']:.2%}")
        print(f"    Sharpe:  {m['sharpe']:.3f}")
        print(f"    Calmar:  {m['calmar']:.3f}")
        print(f"    Minimax: {m['minimax']:.3f}")
        for yr, ret in sorted(m['annual_rets'].items()):
            yc = m['yearly_calmars'].get(yr, float('nan'))
            print(f"    {yr}:   ret={ret:+.2%}  calmar={yc:+.2f}")

    # ── 可视化 ────────────────────────────────────────────────────────
    plot_equity(val_eq, test_eq, csi300_val, csi300_test, best_p)
    plot_yearly_calmar(all_results)

    # ── 保存最优参数 ──────────────────────────────────────────────────
    with open(OUT_DIR / "best_strategy_params_v2.json", "w") as f:
        json.dump({
            "best_params":    asdict(best_p),
            "search_version": "v2_minimax_regime_switching",
            "val_period":     f"{VAL_START}~{VAL_END}",
            "test_period":    f"{TEST_START}~{TEST_END}",
            "val_metrics":    {k: v for k, v in val_m.items()
                               if k not in ('params', 'equity_df')},
            "test_metrics":   {k: v for k, v in test_m.items()
                               if k not in ('params', 'equity_df')},
            "default_test":   {k: v for k, v in default_m.items()
                               if k not in ('params', 'equity_df')},
        }, f, indent=2, default=str)
    print(f"\n  最优参数已保存: {OUT_DIR}/best_strategy_params_v2.json")
    print(f"  总耗时: {time.time()-t0:.1f}s")
    print("═" * 70)


if __name__ == "__main__":
    main()
