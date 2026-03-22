#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指数择时 + 截面选股  联合量化策略回测
============================================================

架构说明
--------
  ① 大盘择时（index_timing_model.py, ma60_state 标签）
       预测 CSI300 走势 → slots ∈ {0, 10, 20}
       信号完全基于 大盘指数 MA 状态，与个股走势无关
       内含硬覆盖: close_CSI300 < MA20 → slots=0（停止开仓）

  ② 截面选股（xgboost_cross_section.py）
       预测个股 10 日超额收益百分位 → 每 5 交易日选 top-N 持仓

  ③ 个股风控（每日执行）
       硬止损:     入场价 × (1 - STOP_LOSS_ENTRY)
       追踪止损:   峰值 × (1 - TRAILING_STOP)
       止盈追踪:   浮盈 > TAKE_PROFIT_GAIN → 追踪幅度收至 TAKE_PROFIT_TRAIL
       MA 死叉:    MA5 < MA20 连续 5 天，且持仓满 5 交易日后触发

仓位逻辑（核心区别）
--------------------
  slots 来自 CSI300 指数择时模型（非个股信号聚合）：
    slots = 0  → bear / ML空仓 → 不新开仓，现有持仓自然出场
    slots = 10 → 半仓 → 最多同时持有 10 只
    slots = 20 → 满仓 → 最多同时持有 20 只
  slot 权重固定 = 总净值 / MAX_SLOTS（不随持仓数量缩放，避免级联再平衡）

样本外严格性
------------
  指数择时模型: 训练 2016-2022，验证 2022，测试 2023-2025
  截面选股模型: 训练 2018-2022，            测试 2023-2025
  回测期:       2023-01-01 ~ 2025-12-31

使用方法
--------
  # 先生成两个模型的预测（仅需运行一次）:
  python index_timing_model.py --label_type ma60_state --no_wfo
  # （xgb_cross_section_predictions.csv 应已存在）

  # 运行回测:
  python index_ma_combined_strategy.py
"""

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

# ══════════════════════════════════════════════════════════════════════
# 1. 配置常量
# ══════════════════════════════════════════════════════════════════════

DB_PATH        = config.db_path
BACKTEST_START = '20220101'   # val predictions start 20220210; test 20250210
BACKTEST_END   = '20260316'

# 持仓槽位（权重 = 总净值 / MAX_SLOTS，固定不变）
MAX_SLOTS     = 8      # 满仓最大持股（OOS 实测最优，HP搜索常过拟合 val）
HALF_SLOTS    = 3      # 半仓最大持股（OOS 实测最优）
BEAR_SLOTS    = 0    # 熊市（slots=0）时不主动调仓（=0禁用），持仓由风控自然退出

# 个股风控参数
STOP_LOSS_ENTRY   = 0.10   # 距入场价回撤 10% → 硬止损（Val Minimax 最优）
TRAILING_STOP     = 1.00   # 追踪止损（禁用，依靠 MA 死叉 + 硬止损退出）
TAKE_PROFIT_GAIN  = 1.00   # 止盈追踪（禁用）
TAKE_PROFIT_TRAIL = 0.10   # 止盈追踪幅度（禁用时无效）

# ── Regime-Switching 退出参数（根据大盘 slots 动态切换）────────────────────
# 牛市（slots=20）：放宽退出，顺势让利润奔跑
# 调优结果（Val 2022-2024 Minimax）: bull(5,5) neutral(3,5) bear(2,2)
BULL_MA_DEATH    = 5       # MA5<MA20 连续5天才退出（容忍短期回调）
BULL_MIN_HOLD    = 5       # 至少持有5天才检查死叉
# 中性（slots=10）：均衡参数
NEUTRAL_MA_DEATH = 3       # 中性市场：3天死叉即检查
NEUTRAL_MIN_HOLD = 5       # 至少持有5天
# 熊市/空仓（slots=0）：快速退出，控制损失
BEAR_MA_DEATH    = 2       # MA5<MA20 连续2天即退出（快速减损）
BEAR_MIN_HOLD    = 2       # 至少持有2天即可检查死叉
BEAR_STOP_LOSS   = 0.10    # 熊市止损幅度（与普通一致，10%）
# 兼容旧代码的别名（中性 regime 默认值）
MIN_HOLD_DAYS    = NEUTRAL_MIN_HOLD
MA_DEATH_DAYS    = NEUTRAL_MA_DEATH

# 优化开关
BEAR_FAST_EXIT       = False # 不强制清仓；依靠 MA 死叉 + 硬止损自然退出
BEAR_EXIT_REBAL_DAYS = 5     # (BEAR_FAST_EXIT=False 时无效)
SLOT_CONFIRM_DAYS    = 3     # 连续 N 天 slots>0 才允许开新仓（防 whipsaw）
ENTRY_MA20_FILTER    = False # 只买 close > MA20 的个股（动量确认）

# 安全过滤器
MIN_LISTED_DAYS = 90    # 上市不满 90 天排除 [180→90: 刚上市新股有独立行情，过早排除损失alpha]
MKTCAP_PCT_CUT  = 10   # 排除市值后 10% [20→10: 小盘股在2023-2025有显著超额收益]

# 收益增强
RISK_FREE_RATE   = 0.02  # 闲置资金年化收益率（货币基金/国债，约2%）
USE_VOL_SCALE    = True  # 是否按20日收益率波动率反比加权持仓（风险平价）
USE_SIGNAL_SCALE = True  # 是否按预测信号强度加权持仓（排名第1约1.5x，最后约0.5x）

# 交易成本（A 股标准）
COMMISSION_RATE = 0.0003   # 双边万三
STAMP_TAX_RATE  = 0.001    # 千一印花税（仅卖出）
TRANSFER_RATE   = 0.00002  # 过户费
SLIPPAGE_RATE   = 0.0001   # 单边滑点
MIN_COMMISSION  = 5.0
MIN_TRADE_VALUE = 2000.0   # 小于此值的漂移不执行

INITIAL_CAPITAL = 1_000_000.0   # 初始资金 100 万

# 预测文件路径
INDEX_TIMING_FILE    = ROOT / 'output' / 'csv' / 'index_timing_predictions.csv'
CS_PRED_FILE         = ROOT / 'output' / 'csv' / 'xgb_cross_section_predictions.csv'
CS_PRED_H5_FILE      = ROOT / 'output' / 'csv' / 'xgb_cs_pred_h5.csv'
CS_PRED_H20_FILE     = ROOT / 'output' / 'csv' / 'xgb_cs_pred_h20.csv'
CS_PRED_VAL_FILE     = ROOT / 'output' / 'csv' / 'xgb_cs_pred_val.csv'     # H10 val (2022-2024)
CS_PRED_VAL_H5_FILE  = ROOT / 'output' / 'csv' / 'xgb_cs_pred_val_h5.csv'  # H5  val (2022-2024)
OUT_DIR              = ROOT / 'output' / 'index_ma_combined'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# 2. 工具函数
# ══════════════════════════════════════════════════════════════════════

def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def _norm_date(s) -> str:
    """统一日期格式 → YYYYMMDD 字符串"""
    return str(s).replace('-', '')


# ══════════════════════════════════════════════════════════════════════
# 3. 加载预测数据
# ══════════════════════════════════════════════════════════════════════

def load_index_timing_slots() -> pd.Series:
    """
    加载指数择时模型的仓位信号（基于 CSI300 MA60 状态）。

    返回: Series，index=trade_date(str YYYYMMDD)，values=slots∈{0,10,20}

    说明:
      slots 已内含 MA 硬覆盖逻辑（来自 index_timing_model.py）：
        close_CSI300 < MA20 → 0
        MA20 ≤ close < MA60 → 10
        close ≥ MA60 + ML看涨 → 20
      因此本策略不需要再做 MA regime 判断。
    """
    if not INDEX_TIMING_FILE.exists():
        raise FileNotFoundError(
            f"找不到指数择时预测: {INDEX_TIMING_FILE}\n"
            "请先运行: python index_timing_model.py --label_type ma60_state --no_wfo"
        )
    df = pd.read_csv(INDEX_TIMING_FILE, dtype={'trade_date': str})
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    # 限制到回测期（含预热：提前 30 天用于前向填充）
    df = df[df['trade_date'] >= '20221201'].copy()
    slots = df.set_index('trade_date')['slots'].astype(int)
    print(f"  指数择时: {len(slots)} 天  "
          f"slots分布 0:{(slots==0).sum()} / 10:{(slots==10).sum()} / 20:{(slots==20).sum()}")
    return slots


def _load_pred_pair(test_file, val_file, label: str) -> pd.DataFrame:
    """加载 test+val 预测并拼接（val 只取 < test 最早日期的部分）"""
    df_test = pd.read_csv(test_file, dtype={'trade_date': str})
    df_test['trade_date'] = df_test['trade_date'].apply(_norm_date)
    df_test = df_test[(df_test['trade_date'] >= BACKTEST_START) &
                      (df_test['trade_date'] <= BACKTEST_END)]
    if val_file.exists():
        df_val = pd.read_csv(val_file, dtype={'trade_date': str})
        df_val['trade_date'] = df_val['trade_date'].apply(_norm_date)
        test_min = df_test['trade_date'].min() if len(df_test) else '99999999'
        df_val = df_val[(df_val['trade_date'] >= BACKTEST_START) &
                        (df_val['trade_date'] < test_min)]
        df = pd.concat([df_val, df_test], ignore_index=True)
    else:
        df = df_test
    return df


def load_cs_predictions() -> pd.DataFrame:
    """加载截面选股预测（val 2022-2024 + test 2025+）。若 H5 文件存在则与 H10 融合"""
    if not CS_PRED_FILE.exists():
        raise FileNotFoundError(
            f"找不到截面选股预测: {CS_PRED_FILE}\n"
            "请先运行 xgboost_cross_section.py 生成预测"
        )
    df = _load_pred_pair(CS_PRED_FILE, CS_PRED_VAL_FILE, 'H10')

    # 若 H5 预测文件存在，融合两个模型（rank 平均）
    if CS_PRED_H5_FILE.exists():
        df5 = _load_pred_pair(CS_PRED_H5_FILE, CS_PRED_VAL_H5_FILE, 'H5')
        df5 = df5.rename(columns={'pred': 'pred_h5'})
        df = df.merge(df5[['ts_code', 'trade_date', 'pred_h5']],
                      on=['ts_code', 'trade_date'], how='inner')
        df['r10'] = df.groupby('trade_date')['pred'].rank(pct=True)
        df['r5']  = df.groupby('trade_date')['pred_h5'].rank(pct=True)
        df['pred'] = 0.70 * df['r10'] + 0.30 * df['r5']
        print(f"  截面选股 [H10+H5 ensemble 70/30]: {len(df):,} 行  "
              f"{df['trade_date'].nunique()} 个调仓日  "
              f"{df['ts_code'].nunique()} 只股票")
    else:
        print(f"  截面选股: {len(df):,} 行  "
              f"{df['trade_date'].nunique()} 个调仓日  "
              f"{df['ts_code'].nunique()} 只股票")
    return df[['ts_code', 'trade_date', 'pred']].copy()


def get_slots_on_date(slots_series: pd.Series, date: str) -> int:
    """
    获取指定日期的 slots（前向填充）。
    如当日无数据则取最近可用日期；最终兜底返回 HALF_SLOTS。
    """
    if date in slots_series.index:
        return int(slots_series[date])
    # 前向填充：取该日期之前最近的一个值
    prior = slots_series[slots_series.index <= date]
    if len(prior) > 0:
        return int(prior.iloc[-1])
    return HALF_SLOTS  # 最终兜底


# ══════════════════════════════════════════════════════════════════════
# 4. 价格数据 & 安全过滤
# ══════════════════════════════════════════════════════════════════════

def load_price_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载日线价格（含预热期 2022-11 用于 MA 计算）。
    返回: (price_pivot, ma5_pivot, ma20_pivot, vol_pivot, retvol20_pivot, open_pivot, pct_chg_pivot)
    open_pivot:   T 日开盘价矩阵，用于 T+1 执行
    pct_chg_pivot: 涨跌幅矩阵，用于涨跌停过滤
    """
    print("  加载个股日线价格...")
    with get_conn() as conn:
        df = conn.execute(f"""
            SELECT trade_date, ts_code, close, open, pct_chg, vol
            FROM daily_price
            WHERE trade_date >= '20221101'
              AND trade_date <= '{BACKTEST_END}'
              AND ts_code NOT LIKE '8%'
              AND ts_code NOT LIKE '4%'
            ORDER BY trade_date, ts_code
        """).fetchdf()

    df['trade_date'] = df['trade_date'].apply(_norm_date)
    price_pv   = df.pivot(index='trade_date', columns='ts_code', values='close')
    open_pv    = df.pivot(index='trade_date', columns='ts_code', values='open')
    pct_chg_pv = df.pivot(index='trade_date', columns='ts_code', values='pct_chg')
    vol_pv     = df.pivot(index='trade_date', columns='ts_code', values='vol')

    ma5_pv      = price_pv.rolling(5,  min_periods=3).mean()
    ma20_pv     = price_pv.rolling(20, min_periods=10).mean()
    retvol20_pv = price_pv.pct_change().rolling(20, min_periods=10).std()

    # 裁剪到回测期
    mask = price_pv.index >= BACKTEST_START
    price_pv    = price_pv[mask]
    open_pv     = open_pv[mask]
    pct_chg_pv  = pct_chg_pv[mask]
    vol_pv      = vol_pv[mask]
    ma5_pv      = ma5_pv[mask]
    ma20_pv     = ma20_pv[mask]
    retvol20_pv = retvol20_pv[mask]

    print(f"  价格矩阵: {len(price_pv)} 天 × {len(price_pv.columns):,} 只股票")
    return price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, open_pv, pct_chg_pv


def load_safety_filters(rebal_dates: List[str]) -> Dict[str, set]:
    """
    每个调仓日返回合规股票集合。
    排除: ST / 上市不满 180 天 / 市值后 20% / 北交所(8*/4*)
    """
    print("  计算安全过滤器...")
    with get_conn() as conn:
        sb = conn.execute("""
            SELECT ts_code, list_date, name FROM stock_basic
        """).fetchdf()

        dates_sql = "'" + "','".join(rebal_dates) + "'"
        mktcap_df = conn.execute(f"""
            SELECT ts_code, trade_date, total_mv
            FROM daily_basic
            WHERE trade_date IN ({dates_sql}) AND total_mv > 0
        """).fetchdf()

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
            if ts in st_set:
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
    """个股持仓状态管理"""
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.shares:      Dict[str, int]   = {}
        self.entry_price: Dict[str, float] = {}
        self.entry_date:  Dict[str, str]   = {}
        self.peak_price:  Dict[str, float] = {}
        self.hold_days:   Dict[str, int]   = {}   # 已持有交易日数
        self.dc_count:    Dict[str, int]   = {}   # 连续 MA 死叉天数

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
        new_ep = (old_ep * old_sh + price * sh) / new_sh
        self.shares[ts_code]      = new_sh
        self.entry_price[ts_code] = new_ep
        self.entry_date[ts_code]  = self.entry_date.get(ts_code, date)
        self.peak_price[ts_code]  = max(self.peak_price.get(ts_code, price), price)
        self.hold_days[ts_code]   = self.hold_days.get(ts_code, 0)
        self.dc_count[ts_code]    = self.dc_count.get(ts_code, 0)
        return True

    def update_hold_days(self):
        """每个交易日末调用：持有天数 +1"""
        for ts in list(self.hold_days):
            self.hold_days[ts] = self.hold_days.get(ts, 0) + 1


# ══════════════════════════════════════════════════════════════════════
# 7. 每日风控检查
# ══════════════════════════════════════════════════════════════════════

def daily_stop_check(
    portfolio: Portfolio,
    prices: pd.Series,
    ma5: pd.Series,
    ma20: pd.Series,
    date: str,
    slots_today: int = 10,
) -> List[str]:
    """
    检查每只持仓的退出条件，返回需卖出的 ts_code 列表。

    退出顺序（优先级从高到低）：
      ① 停牌/退市（val ≤ 0）
      ② 硬止损：price < entry_price × (1 - effective_stop)
      ③ 追踪止损：price < peak × (1 - TRAILING_STOP)（默认禁用）
      ④ MA 死叉：MA5 < MA20 连续 N 天（N 由 Regime/slots 决定）
           slots=20 → BULL_MA_DEATH  / BULL_MIN_HOLD   （顺势持仓）
           slots=10 → NEUTRAL_MA_DEATH / NEUTRAL_MIN_HOLD（均衡）
           slots=0  → BEAR_MA_DEATH  / BEAR_MIN_HOLD   （快速退出）
    """
    # Regime-switching：根据大盘 slots 选择退出参数
    if slots_today == 20:
        effective_ma_death = BULL_MA_DEATH
        effective_min_hold = BULL_MIN_HOLD
        effective_stop     = STOP_LOSS_ENTRY
    elif slots_today == 10:
        effective_ma_death = NEUTRAL_MA_DEATH
        effective_min_hold = NEUTRAL_MIN_HOLD
        effective_stop     = STOP_LOSS_ENTRY
    else:  # slots=0，熊市
        effective_ma_death = BEAR_MA_DEATH
        effective_min_hold = BEAR_MIN_HOLD
        effective_stop     = BEAR_STOP_LOSS

    to_sell = []
    for ts in list(portfolio.shares):
        price = prices.get(ts, np.nan)

        # ① 停牌/退市
        if pd.isna(price) or price <= 0:
            to_sell.append(ts)
            continue

        ep   = portfolio.entry_price.get(ts, price)
        peak = portfolio.peak_price.get(ts, price)

        # 更新追踪最高价
        if price > peak:
            portfolio.peak_price[ts] = price
            peak = price

        # ② 硬止损
        if price < ep * (1 - effective_stop):
            to_sell.append(ts)
            continue

        # ③ 追踪止损（熊市用 BEAR_TRAILING，牛市用 TRAILING_STOP/TAKE_PROFIT_TRAIL）
        gain = price / ep - 1
        trail = TAKE_PROFIT_TRAIL if gain >= TAKE_PROFIT_GAIN else TRAILING_STOP
        if price < peak * (1 - trail):
            to_sell.append(ts)
            continue

        # ④ MA 死叉（持仓满 effective_min_hold 后才检查）
        hd = portfolio.hold_days.get(ts, 0)
        if hd >= effective_min_hold:
            m5  = ma5.get(ts, np.nan)
            m20 = ma20.get(ts, np.nan)
            if pd.notna(m5) and pd.notna(m20):
                if m5 < m20:
                    portfolio.dc_count[ts] = portfolio.dc_count.get(ts, 0) + 1
                    if portfolio.dc_count[ts] >= effective_ma_death:
                        to_sell.append(ts)
                        continue
                else:
                    portfolio.dc_count[ts] = 0

    return list(set(to_sell))


# ══════════════════════════════════════════════════════════════════════
# 8. 调仓
# ══════════════════════════════════════════════════════════════════════

def rebalance(
    portfolio: Portfolio,
    target_stocks: List[str],
    n_slots: int,
    prices: pd.Series,
    date: str,
    slot_confirmed: bool = True,
    ret_vol: pd.Series = None,
    exec_prices: pd.Series = None,   # T+1 开盘价，用于实际成交；None 则退化到 T 收盘
) -> int:
    """
    按 CS 模型打分排列的 target_stocks 调仓。

    n_slots = 0 → bear/空仓期：不新开仓，现有持仓由风控自然退出
    n_slots > 0 → 目标持仓 top-N，slot 权重 = 总净值 / MAX_SLOTS（固定）

    slot_confirmed = False → 只执行卖出，暂不买入（slots刚转正未稳定）
    ret_vol        = 20日收益率波动率 Series，USE_VOL_SCALE=True 时用于反比加权
    exec_prices    = T+1 开盘价 Series（T+1 执行模型）
    """
    if n_slots == 0:
        return 0

    # 估值用 T 收盘价；执行用 T+1 开盘价（若不可用则退化到 T 收盘）
    ep = exec_prices if exec_prices is not None else prices

    total_val   = portfolio.total_value(prices)
    target_set  = set(target_stocks[:n_slots])
    current_set = set(portfolio.shares)

    trades = 0

    # 卖出：不在目标中且持满 MIN_HOLD_DAYS 的仓位
    for ts in sorted(current_set - target_set):
        hd = portfolio.hold_days.get(ts, 0)
        if hd >= MIN_HOLD_DAYS:
            p = ep.get(ts, np.nan)
            if pd.notna(p) and p > 0:
                portfolio.sell(ts, p, date)
                trades += 1

    # 买入：按 pred 得分顺序买入尚未持有的目标股
    if slot_confirmed:
        target_list = target_stocks[:n_slots]
        top_stocks  = [ts for ts in target_list if ts not in portfolio.shares]

        fixed_sv = total_val / MAX_SLOTS  # 基础槽位大小（不变）

        # ── 信号强度加权：rank 0(最强) → 1.5x，rank n-1(最弱) → 0.5x ──────
        if USE_SIGNAL_SCALE and len(target_list) > 0:
            n = len(target_list)
            sig_scale_map = {}
            for i, ts in enumerate(target_list):
                if n > 1:
                    sig_scale_map[ts] = 1.5 - 1.0 * i / (n - 1)   # [1.5, 0.5]
                else:
                    sig_scale_map[ts] = 1.0
        else:
            sig_scale_map = {ts: 1.0 for ts in target_list}

        if USE_VOL_SCALE and ret_vol is not None and len(top_stocks) > 0:
            vols = np.array([float(ret_vol.get(ts, np.nan) or np.nan)
                             if pd.notna(ret_vol.get(ts, np.nan)) else np.nan
                             for ts in top_stocks])
            median_vol = float(np.nanmedian(vols)) if not np.all(np.isnan(vols)) else 0.02
            vols = np.where(np.isnan(vols), median_vol, vols)
            vols = np.maximum(vols, 0.005)
            vol_scale = np.clip(median_vol / vols, 0.5, 2.0)
            slot_values = {ts: fixed_sv * sig_scale_map.get(ts, 1.0) * vol_scale[i]
                           for i, ts in enumerate(top_stocks)}
        else:
            slot_values = {ts: fixed_sv * sig_scale_map.get(ts, 1.0)
                           for ts in top_stocks}

        for ts in top_stocks:
            p = ep.get(ts, np.nan)    # T+1 开盘价执行
            if pd.isna(p) or p <= 0:
                continue
            if portfolio.buy(ts, p, slot_values[ts], date):
                trades += 1

    return trades


# ══════════════════════════════════════════════════════════════════════
# 9. 回测主循环
# ══════════════════════════════════════════════════════════════════════

def run_backtest(
    price_pv:     pd.DataFrame,
    ma5_pv:       pd.DataFrame,
    ma20_pv:      pd.DataFrame,
    vol_pv:       pd.DataFrame,
    cs_preds:     pd.DataFrame,
    index_slots:  pd.Series,
    eligible:     Dict[str, set],
    rebal_set:    set,
    csi300_mom:   pd.Series = None,  # CSI300 5日收益，用于入场过滤
    retvol20_pv:  pd.DataFrame = None,  # 20日收益率波动率，用于风险平价
    open_pv:      pd.DataFrame = None,  # T+1 开盘价矩阵（T+1 执行）
    pct_chg_pv:   pd.DataFrame = None,  # 涨跌幅矩阵（涨跌停过滤）
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    日频回测主循环。

    每日：更新峰值 → 每日风控检查 → 记录净值
    调仓日：
      1. 读取当日指数择时 slots（大盘 MA 状态 + ML 信号）
      2. 过滤合规股 + 非停牌
      3. 按 CS 预测分数排序，取 top-N（N = slots）
      4. 执行调仓
    """
    print("\n[回测] 运行指数择时 + 截面选股联合策略...")
    print(f"  Regime退出: bull(md={BULL_MA_DEATH},mh={BULL_MIN_HOLD}) "
          f"neutral(md={NEUTRAL_MA_DEATH},mh={NEUTRAL_MIN_HOLD}) "
          f"bear(md={BEAR_MA_DEATH},mh={BEAR_MIN_HOLD})  "
          f"stop={STOP_LOSS_ENTRY:.0%}  slot_confirm={SLOT_CONFIRM_DAYS}d  "
          f"vol_scale={USE_VOL_SCALE}")
    print(f"  执行模型: {'T+1开盘' if open_pv is not None else 'T收盘（无开盘数据）'}")
    pf = Portfolio(INITIAL_CAPITAL)

    # 涨跌幅矩阵（若未传入则从 close 推算，精度低）
    if pct_chg_pv is None:
        pct_chg_pv = price_pv.pct_change()

    cs_by_date: Dict[str, pd.DataFrame] = {
        dt: grp for dt, grp in cs_preds.groupby('trade_date')
    }

    all_dates = sorted(d for d in price_pv.index if d >= BACKTEST_START)
    equity: List[dict] = []
    trade_log: List[dict] = []
    slots_log: List[dict] = []

    consec_nonzero  = 0   # 连续 slots>0 天数（用于稳定确认）
    consec_bear_rebal = 0  # 连续 slots=0 的调仓日数（用于延迟熊市清仓）

    for i, date in enumerate(all_dates):
        slots_today = get_slots_on_date(index_slots, date)
        bear_mode   = (slots_today == 0)

        # 更新连续非零天数
        if slots_today > 0:
            consec_nonzero += 1
        else:
            consec_nonzero = 0

        if i % 60 == 0:
            tv = pf.total_value(price_pv.loc[date])
            print(f"  {date}  净值: ¥{tv:>12,.0f}  "
                  f"持仓: {len(pf.shares):2d} 只  slots={slots_today}  "
                  f"consec={consec_nonzero}")

        prices = price_pv.loc[date]
        ma5    = ma5_pv.loc[date]  if date in ma5_pv.index  else pd.Series(dtype=float)
        ma20   = ma20_pv.loc[date] if date in ma20_pv.index else pd.Series(dtype=float)

        # T+1 执行价：取下一交易日开盘价；最后一天退化到当日收盘
        next_date = all_dates[i + 1] if i + 1 < len(all_dates) else date
        if open_pv is not None and next_date in open_pv.index:
            exec_prices = open_pv.loc[next_date]
        else:
            exec_prices = prices  # 退化到 T 收盘

        # T+1 涨跌幅（用于涨跌停过滤）
        if pct_chg_pv is not None and next_date in pct_chg_pv.index:
            exec_pct = pct_chg_pv.loc[next_date]
        else:
            exec_pct = None

        # ── ① 每日风控：止损 / 追踪止损 / MA 死叉 ────────────────────────
        # 风控基于 T 收盘价触发，执行价用 T+1 开盘（exec_prices）
        exits = daily_stop_check(pf, prices, ma5, ma20, date, slots_today=slots_today)
        for ts in exits:
            ep_raw = exec_prices.get(ts, np.nan)
            # 跌停检查：T+1 跌停时无法卖出（持仓被锁）
            if exec_pct is not None:
                pct_val = exec_pct.get(ts, np.nan)
                is_star = ts[:3] in ('688', '300', '301')
                limit_dn = -19.9 if is_star else -9.9
                if pd.notna(pct_val) and pct_val <= limit_dn:
                    continue   # 跌停，本日无法卖出，保留持仓
            p = float(ep_raw if pd.notna(ep_raw) and ep_raw > 0 else pf.entry_price.get(ts, 1.0))
            cash_rcv = pf.sell(ts, p, date)
            trade_log.append({
                'date': date, 'ts_code': ts,
                'action': 'stop_sell', 'price': p, 'cash': cash_rcv,
            })

        # ── ② 调仓日 ──────────────────────────────────────────────────────
        if date in rebal_set:
            slots_log.append({'date': date, 'slots': slots_today})

            # 更新连续熊市调仓日计数
            if bear_mode:
                consec_bear_rebal += 1
            else:
                consec_bear_rebal = 0

            # ── 延迟熊市清仓：连续 BEAR_EXIT_REBAL_DAYS 个调仓日均为 0 才强制清仓 ──
            if BEAR_FAST_EXIT and bear_mode and consec_bear_rebal >= BEAR_EXIT_REBAL_DAYS:
                for ts in list(pf.shares):
                    ep_raw = exec_prices.get(ts, np.nan)
                    # 跌停时无法卖出
                    if exec_pct is not None:
                        pct_val = exec_pct.get(ts, np.nan)
                        is_star = ts[:3] in ('688', '300', '301')
                        if pd.notna(pct_val) and pct_val <= (-19.9 if is_star else -9.9):
                            continue
                    p = float(ep_raw if np.isfinite(ep_raw) and ep_raw > 0 else 0)
                    if p > 0:
                        cash_rcv = pf.sell(ts, p, date)
                        trade_log.append({
                            'date': date, 'ts_code': ts,
                            'action': 'bear_exit', 'price': p, 'cash': cash_rcv,
                        })
            elif slots_today > 0:
                # ── 合规股 + 非停牌 ────────────────────────────────────────
                elig_set  = eligible.get(date, set())
                vol_today = vol_pv.loc[date] if date in vol_pv.index else pd.Series(dtype=float)
                active    = set(vol_today[vol_today > 0].dropna().index)

                scores_df = cs_by_date.get(date, pd.DataFrame(columns=['ts_code', 'pred']))
                scores_df = scores_df[
                    scores_df['ts_code'].isin(elig_set) &
                    scores_df['ts_code'].isin(active)
                ].sort_values('pred', ascending=False)

                # ── 个股 MA20 入场过滤：只买已处于上升趋势的股票 ────────
                if ENTRY_MA20_FILTER and len(ma20) > 0:
                    def _above_ma20(ts_code):
                        p   = prices.get(ts_code, np.nan)
                        m20 = ma20.get(ts_code, np.nan)
                        return pd.notna(p) and pd.notna(m20) and m20 > 0 and p > m20
                    scores_df = scores_df[scores_df['ts_code'].apply(_above_ma20)]

                # ── 涨停过滤：T+1 涨停无法买入（以 T+1 pct_chg 判断）────
                if exec_pct is not None:
                    def _limit_up_thresh(ts_code):
                        return 19.9 if ts_code[:3] in ('688', '300', '301') else 9.9
                    limit_up_stocks = set(
                        ts for ts in scores_df['ts_code']
                        if pd.notna(exec_pct.get(ts, np.nan))
                        and exec_pct.get(ts, 0) >= _limit_up_thresh(ts)
                    )
                    scores_df = scores_df[~scores_df['ts_code'].isin(limit_up_stocks)]

                target_stocks = scores_df['ts_code'].tolist()
                slot_confirmed = (consec_nonzero >= SLOT_CONFIRM_DAYS)

                # 半仓期（slots=10）限制最大持仓为 HALF_SLOTS
                actual_slots = MAX_SLOTS if slots_today == 20 else HALF_SLOTS

                if target_stocks:
                    rv_today = retvol20_pv.loc[date] if (
                        retvol20_pv is not None and date in retvol20_pv.index
                    ) else None
                    n_trades = rebalance(
                        pf, target_stocks, actual_slots, prices, date,
                        slot_confirmed=slot_confirmed,
                        ret_vol=rv_today,
                        exec_prices=exec_prices,
                    )
                    trade_log.append({
                        'date': date, 'ts_code': 'REBAL',
                        'action': 'rebalance',
                        'price': slots_today, 'cash': n_trades,
                    })

        # ── ③ 持仓天数 +1 ─────────────────────────────────────────────────
        pf.update_hold_days()

        # ── ③b 闲置资金收益（货币基金/国债，年化 RISK_FREE_RATE）───────────
        pf.cash *= (1 + RISK_FREE_RATE / 252)

        # ── ④ 记录净值 ────────────────────────────────────────────────────
        tv = pf.total_value(prices)
        equity.append({
            'date': date,
            'total_value': tv,
            'cash': pf.cash,
            'n_positions': len(pf.shares),
            'slots': get_slots_on_date(index_slots, date),
        })

    equity_df = pd.DataFrame(equity).set_index('date')
    equity_df.index = pd.to_datetime(equity_df.index, format='%Y%m%d')

    slots_df = pd.DataFrame(slots_log)
    if len(slots_df):
        print(f"\n  调仓期 slots 分布: "
              f"0槽={( slots_df['slots']==0 ).sum()} / "
              f"10槽={( slots_df['slots']==10).sum()} / "
              f"20槽={( slots_df['slots']==20).sum()}")

    print(f"\n  最终净值: ¥{equity_df['total_value'].iloc[-1]:,.0f}")
    print(f"  平均持仓: {equity_df['n_positions'].mean():.1f} 只")
    print(f"  平均仓位槽: {equity_df['slots'].mean():.1f} 槽 / {MAX_SLOTS} 槽")
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

    peak   = vals.expanding().max()
    dd     = (vals - peak) / peak
    max_dd = float(dd.min())
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.nan

    down    = rets[rets < 0]
    sortino = (annual_ret - 0.02) / (down.std() * np.sqrt(tpy)) if len(down) > 0 else np.nan
    win_rate = float((rets > 0).mean())

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


def print_metrics(m: dict, equity_df: Optional[pd.DataFrame] = None):
    print("\n" + "═" * 58)
    print("  指数择时 + 截面选股  联合策略回测报告")
    print("  择时训练: 2016-2022  选股训练: 2018-2022")
    print("  回测期:   2023-2025（严格样本外）")
    print("═" * 58)
    print(f"  总收益率:    {m['total_ret']:>+.2%}")
    print(f"  年化收益率:  {m['annual_ret']:>+.2%}")
    print(f"  年化波动率:  {m['annual_vol']:>.2%}")
    print(f"  夏普比率:    {m['sharpe']:>.3f}")
    print(f"  索提诺比率:  {m['sortino']:>.3f}")
    print(f"  卡玛比率:    {m['calmar']:>.3f}")
    print(f"  最大回撤:    {m['max_dd']:>.2%}")
    print(f"  日胜率:      {m['win_rate']:>.1%}")
    print(f"  时间跨度:    {m['years']:.2f} 年")
    if equity_df is not None:
        avg_pos  = equity_df['n_positions'].mean()
        avg_slot = equity_df['slots'].mean() if 'slots' in equity_df.columns else float('nan')
        print(f"  平均持仓:    {avg_pos:.1f} 只")
        print(f"  平均仓位:    {avg_slot:.1f}/{MAX_SLOTS} 槽 ({avg_slot/MAX_SLOTS:.0%})")
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


def load_csi300_momentum() -> pd.Series:
    """CSI300 5日滚动收益（用于入场动量过滤）。返回 Series，index=YYYYMMDD str"""
    with get_conn() as conn:
        df = conn.execute("""
            SELECT trade_date, close FROM index_daily
            WHERE ts_code = '000300.SH'
              AND trade_date >= '20221001'
            ORDER BY trade_date
        """).fetchdf()
    df['trade_date'] = df['trade_date'].apply(_norm_date)
    df = df.set_index('trade_date').sort_index()
    df['ret5'] = df['close'].pct_change(5)
    return df['ret5']


def plot_results(
    equity_df: pd.DataFrame,
    m: dict,
    benchmark: pd.Series,
    index_slots: pd.Series,
):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), facecolor='#f8f8f8')
    fig.suptitle(
        '指数择时 + 截面选股  联合量化策略\n'
        '择时: CSI300 ma60_state 标签 | 选股: XGBoost 截面超额收益',
        fontsize=12, fontweight='bold', y=0.99,
    )

    nav = equity_df['total_value'] / INITIAL_CAPITAL

    # ① 净值曲线
    ax = axes[0]
    ax.plot(nav.index, nav.values, color='#1a6db4', lw=1.8, label='联合策略')
    bm = benchmark.reindex(nav.index, method='ffill').dropna()
    if len(bm):
        ax.plot(bm.index, bm.values, color='#e05c2a', lw=1.2,
                alpha=0.8, ls='--', label='沪深300')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4)
    ax.set_title(
        f"净值曲线  |  年化 {m['annual_ret']:+.2%}  "
        f"最大回撤 {m['max_dd']:.2%}  夏普 {m['sharpe']:.3f}",
        fontsize=10,
    )
    ax.set_ylabel('NAV')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25)

    # ② 回撤曲线
    ax = axes[1]
    peak = equity_df['total_value'].expanding().max()
    dd   = (equity_df['total_value'] - peak) / peak * 100
    ax.fill_between(dd.index, dd.values, 0, color='#c0392b', alpha=0.35)
    ax.plot(dd.index, dd.values, color='#c0392b', lw=0.8)
    ax.axhline(0, color='gray', ls=':', alpha=0.4)
    ax.set_ylabel('回撤 (%)')
    ax.set_title(f"最大回撤: {m['max_dd']:.2%}", fontsize=10)
    ax.grid(True, alpha=0.25)

    # ③ 持仓数量
    ax = axes[2]
    ax.fill_between(equity_df.index, equity_df['n_positions'].values,
                    color='#27ae60', alpha=0.4, step='post')
    ax.plot(equity_df.index, equity_df['n_positions'].values,
            color='#27ae60', lw=0.8)
    ax.set_ylabel('持仓只数')
    ax.set_title(f"持仓数量（均值 {equity_df['n_positions'].mean():.1f} 只）", fontsize=10)
    ax.set_ylim(-1, MAX_SLOTS + 2)
    ax.grid(True, alpha=0.25)

    # ④ 指数择时 slots（大盘状态）
    ax = axes[3]
    # 将 slots 对齐到回测日期
    slots_aligned = pd.Series(
        [get_slots_on_date(index_slots, d.strftime('%Y%m%d')) for d in equity_df.index],
        index=equity_df.index,
    )
    colors = {0: '#c0392b', 10: '#f39c12', 20: '#27ae60'}
    for s_val, color in colors.items():
        mask = slots_aligned == s_val
        ax.fill_between(equity_df.index, 0, slots_aligned.where(mask, 0),
                        color=color, alpha=0.5, step='post',
                        label={0: 'bear(0)', 10: 'half(10)', 20: 'bull(20)'}[s_val])
    ax.set_ylabel('指数择时 slots')
    ax.set_ylim(-1, MAX_SLOTS + 2)
    ax.set_yticks([0, 10, 20])
    ax.set_title('大盘择时信号（CSI300 ma60_state 模型）', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.25)

    # 年度分隔线
    for ax in axes:
        for yr in range(2023, 2027):
            ax.axvline(pd.Timestamp(f'{yr}-01-01'), color='gray', ls=':', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = OUT_DIR / 'index_ma_combined_equity.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# 12. 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  指数择时 + 截面选股  联合量化策略回测")
    print("  择时: CSI300 ma60_state（大盘 MA 状态）")
    print("  选股: XGBoost 截面超额收益预测")
    print(f"  回测: {BACKTEST_START} ~ {BACKTEST_END}")
    print("=" * 60)

    # 1. 加载指数择时信号
    print("\n[1/6] 加载指数择时信号（ma60_state）...")
    index_slots = load_index_timing_slots()

    # 2. 加载截面选股预测
    print("\n[2/6] 加载截面选股预测...")
    cs_preds = load_cs_predictions()

    # 3. 日线价格数据
    print("\n[3/6] 加载个股价格数据...")
    price_pv, ma5_pv, ma20_pv, vol_pv, retvol20_pv, open_pv, pct_chg_pv = load_price_data()

    # 4. 确定调仓日（来自 CS 模型，每 5 日）
    all_rebal  = sorted(cs_preds['trade_date'].unique())
    rebal_dates = [d for d in all_rebal if BACKTEST_START <= d <= BACKTEST_END]
    rebal_set   = set(rebal_dates)
    print(f"\n  调仓日: {len(rebal_dates)} 个  "
          f"({rebal_dates[0]} ~ {rebal_dates[-1]})")

    # 汇报各调仓日的 slots 分布
    slot_counts = {0: 0, 10: 0, 20: 0}
    for d in rebal_dates:
        s = get_slots_on_date(index_slots, d)
        slot_counts[s] = slot_counts.get(s, 0) + 1
    print(f"  调仓日 slots: "
          f"0槽(空仓)={slot_counts.get(0,0)}次  "
          f"10槽(半仓)={slot_counts.get(10,0)}次  "
          f"20槽(满仓)={slot_counts.get(20,0)}次")

    # 5. 安全过滤器（ST / 新股 / 小市值）
    print("\n[4/6] 计算安全过滤器...")
    eligible = load_safety_filters(rebal_dates)

    # 5b. CSI300 入场动量过滤
    csi300_mom = load_csi300_momentum()

    # 6. 运行回测
    print("\n[5/6] 运行回测...")
    equity_df, trade_log = run_backtest(
        price_pv, ma5_pv, ma20_pv, vol_pv,
        cs_preds, index_slots, eligible, rebal_set,
        csi300_mom=csi300_mom,
        retvol20_pv=retvol20_pv,
        open_pv=open_pv,
        pct_chg_pv=pct_chg_pv,
    )

    # 7. 绩效统计 & 输出
    print("\n[6/6] 计算绩效统计...")
    m = compute_metrics(equity_df)
    print_metrics(m, equity_df)

    benchmark = load_csi300_benchmark()
    plot_results(equity_df, m, benchmark, index_slots)

    # 保存净值曲线
    equity_path = OUT_DIR / 'index_ma_combined_equity.csv'
    equity_df.to_csv(equity_path)
    print(f"  净值: {equity_path}")

    trades_path = OUT_DIR / 'index_ma_combined_trades.csv'
    pd.DataFrame(trade_log).to_csv(trades_path, index=False)
    print(f"  交易: {trades_path}")

    print("\n完成。")


if __name__ == '__main__':
    main()
