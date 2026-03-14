#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日 Inference 脚本 — 指数择时 + 截面选股联合策略
=====================================================

打印当前策略的完整推理链：
  [1] 数据库状态      — 各关键表最新数据日期
  [2] 大盘择时模型    — CSI300 MA 状态 + 时序模型最近 pred_prob
  [3] 截面选股模型    — 最新截面日 Top-N 股票预测详情
  [4] 联合策略建议    — 根据当前 slots + CS 排名生成持仓建议
  [5] 最近成交记录    — 从 index_ma_combined_trades.csv
  [6] 数据新鲜度      — 预测 CSV vs DB 最新日期对比

用法:
  python daily_inference.py                   # 使用默认配置
  python daily_inference.py --top_n 10        # 展示 Top-10 股票
  python daily_inference.py --date 20260101   # 指定截面日（默认最新）
"""

import sys
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from data_collect.config import config

# ══════════════════════════════════════════════════════════════════════
# 配置（与 index_ma_combined_strategy.py 对齐）
# ══════════════════════════════════════════════════════════════════════
DB_PATH   = config.db_path
MAX_SLOTS = 8
SLOT_CONFIRM_DAYS = 3    # 连续 N 天 slots>0 才建议开新仓
MIN_LISTED_DAYS   = 180
MKTCAP_PCT_CUT    = 20   # 排除市值后 20%

INDEX_TIMING_FILE = ROOT / 'output' / 'csv' / 'index_timing_predictions.csv'
CS_PRED_FILE      = ROOT / 'output' / 'csv' / 'xgb_cross_section_predictions.csv'
TRADES_FILE       = ROOT / 'output' / 'index_ma_combined' / 'index_ma_combined_trades.csv'

DIVIDER = '═' * 72


# ══════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)


def norm_date(s) -> str:
    return str(s).replace('-', '')


def fmt_pct(v, decimals=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  N/A  '
    sign = '+' if v >= 0 else ''
    return f"{sign}{v * 100:.{decimals}f}%"


def fmt_num(v, decimals=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f"{v:.{decimals}f}"


def bar(val: float, lo: float = 0.0, hi: float = 1.0, width: int = 20) -> str:
    """水平进度条，val 归一化到 [lo, hi]"""
    t = np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)
    filled = int(t * width)
    return '█' * filled + '░' * (width - filled)


# ══════════════════════════════════════════════════════════════════════
# [1] 数据库状态
# ══════════════════════════════════════════════════════════════════════
def section_db_status():
    print(f"\n{DIVIDER}")
    print("  [1] 数据库状态")
    print(DIVIDER)
    with get_conn() as conn:
        tables = {
            'daily_price':      "SELECT MAX(trade_date) FROM daily_price",
            'index_daily':      "SELECT MAX(trade_date) FROM index_daily",
            'daily_basic':      "SELECT MAX(trade_date) FROM daily_basic",
            'moneyflow':        "SELECT MAX(trade_date) FROM moneyflow",
            'income_statement': "SELECT MAX(ann_date) FROM income_statement",
            'fina_indicator':   "SELECT MAX(ann_date) FROM fina_indicator",
        }
        for tbl, sql in tables.items():
            try:
                latest = conn.execute(sql).fetchone()[0]
                print(f"  {tbl:<22}  最新: {latest}")
            except Exception as e:
                print(f"  {tbl:<22}  查询失败: {e}")
    print()


# ══════════════════════════════════════════════════════════════════════
# [2] 大盘择时模型
# ══════════════════════════════════════════════════════════════════════
def _load_csi300_ma(conn) -> pd.DataFrame:
    """从 index_daily 计算 CSI300 MA 序列（含 MA20/MA60/MA250）"""
    df = conn.execute("""
        SELECT trade_date, close
        FROM index_daily
        WHERE ts_code = '000300.SH'
        ORDER BY trade_date
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


def _ma_state(row) -> Tuple[str, int]:
    """根据 CSI300 价格相对 MA 判断市场状态 → (label, raw_slots)"""
    c, ma20, ma60 = row['close'], row['ma20'], row['ma60']
    if pd.isna(ma20) or pd.isna(ma60):
        return 'unknown', 10
    if c < ma20:
        return '熊市 (close < MA20)', 0
    if c < ma60:
        return '中性 (MA20 ≤ close < MA60)', 10
    return '牛市 (close ≥ MA60)', 20


def section_timing(display_days: int = 20):
    print(f"\n{DIVIDER}")
    print("  [2] 大盘择时模型 — CSI300 MA 状态 + 时序模型信号")
    print(DIVIDER)

    # ── (a) 从 DB 计算最新 MA 状态 ──────────────────────────────────
    with get_conn() as conn:
        ma_df = _load_csi300_ma(conn)

    latest_date = ma_df.index[-1]
    latest      = ma_df.loc[latest_date]
    state_label, raw_slots = _ma_state(latest)

    print(f"\n  CSI300 最新数据日期: {latest_date}")
    print(f"  {'收盘价':10}  {latest['close']:.2f}")
    print(f"  {'MA20':10}  {latest['ma20']:.2f}   "
          f"偏离 {fmt_pct(latest['close']/latest['ma20']-1)}")
    print(f"  {'MA60':10}  {latest['ma60']:.2f}   "
          f"偏离 {fmt_pct(latest['close']/latest['ma60']-1)}")
    print(f"  {'MA250':10}  {latest['ma250']:.2f}   "
          f"偏离 {fmt_pct(latest['close']/latest['ma250']-1)}")
    print(f"\n  ▶ MA 状态: {state_label}  →  基础 slots = {raw_slots}")
    print(f"  ▶ 1日涨跌: {fmt_pct(latest['ret_1d'])}   "
          f"5日: {fmt_pct(latest['ret_5d'])}   "
          f"20日: {fmt_pct(latest['ret_20d'])}")

    # ── (b) 近期 MA 状态明细（最近 N 个交易日）────────────────────────
    print(f"\n  近 {display_days} 个交易日 CSI300 MA 状态:")
    print(f"  {'日期':10}  {'收盘':>8}  {'MA20':>8}  {'MA60':>8}  {'状态':>6}  {'Slots':>6}")
    recent = ma_df.tail(display_days)
    for dt, row in recent.iterrows():
        sl, ss = _ma_state(row)
        symbol = '▲' if ss == 20 else ('─' if ss == 10 else '▼')
        print(f"  {dt}  {row['close']:8.2f}  {row['ma20']:8.2f}  "
              f"{row['ma60']:8.2f}  {symbol:>6}  {ss:>6}")

    # ── (c) SLOT_CONFIRM_DAYS 确认状态 ────────────────────────────────
    # 统计最近 SLOT_CONFIRM_DAYS 天是否连续 slots>0
    recent_states = [_ma_state(r)[1] for _, r in recent.tail(SLOT_CONFIRM_DAYS).iterrows()]
    consec_bull = all(s > 0 for s in recent_states)
    print(f"\n  SLOT_CONFIRM_DAYS={SLOT_CONFIRM_DAYS}: "
          f"最近{SLOT_CONFIRM_DAYS}天 slots={recent_states}  "
          f"→ {'✓ 可开新仓' if consec_bull else '✗ 等待确认，暂不开新仓'}")

    # ── (d) 时序模型 pred_prob（从 CSV）────────────────────────────────
    if INDEX_TIMING_FILE.exists():
        timing_df = pd.read_csv(INDEX_TIMING_FILE, dtype={'trade_date': str})
        timing_df['trade_date'] = timing_df['trade_date'].str.replace('-', '')
        timing_df = timing_df.sort_values('trade_date')

        last_timing_date = timing_df['trade_date'].iloc[-1]
        print(f"\n  时序模型预测（CSV）最新日期: {last_timing_date}")
        print(f"  近 {display_days} 天 pred_prob & slots:")
        print(f"  {'日期':10}  {'pred_prob':>10}  {'slots':>6}  {'概率条':}")
        recent_t = timing_df.tail(display_days)
        for _, row in recent_t.iterrows():
            prob = row['pred_prob']
            sl   = int(row['slots'])
            print(f"  {row['trade_date']}  {prob:10.4f}  {sl:6d}  "
                  f"{bar(prob, 0.3, 0.9, 24)}")

        # 统计
        last_30 = timing_df.tail(30)
        print(f"\n  近30天槽位分布: "
              f"0槽={( last_30['slots']==0).sum()}天  "
              f"10槽={(last_30['slots']==10).sum()}天  "
              f"20槽={(last_30['slots']==20).sum()}天")
    else:
        print(f"\n  [!] 未找到时序模型预测文件: {INDEX_TIMING_FILE}")
        print("      请先运行: python index_timing_model.py --label_type ma60_state --no_wfo")

    return raw_slots, consec_bull


# ══════════════════════════════════════════════════════════════════════
# [3] 截面选股模型
# ══════════════════════════════════════════════════════════════════════
def _load_eligible_today(conn, date: str) -> set:
    """合规股票集合（非ST、上市满180天、市值非后20%、非北交所）"""
    sb = conn.execute(
        "SELECT ts_code, list_date, name FROM stock_basic"
    ).fetchdf()
    sb['is_st']    = sb['name'].str.contains('ST', na=False)
    sb['list_date'] = sb['list_date'].astype(str).str.replace('-', '')
    st_set          = set(sb.loc[sb['is_st'], 'ts_code'])
    list_date_map   = dict(zip(sb['ts_code'], sb['list_date']))

    mktcap = conn.execute(f"""
        SELECT ts_code, total_mv
        FROM daily_basic
        WHERE trade_date = '{date}' AND total_mv > 0
    """).fetchdf()

    if mktcap.empty:
        return set()

    cutoff = np.percentile(mktcap['total_mv'].values, MKTCAP_PCT_CUT)
    eligible = set()
    for _, row in mktcap.iterrows():
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
        eligible.add(ts)
    return eligible


def _load_stock_meta(conn) -> pd.DataFrame:
    """加载股票基础信息（代码→名称、行业）"""
    df = conn.execute("""
        SELECT ts_code, name, industry
        FROM stock_basic
    """).fetchdf()
    return df.set_index('ts_code')


def _load_latest_prices(conn, ts_codes: List[str], latest_price_date: str) -> pd.DataFrame:
    """获取指定股票的最近价格、MA5、MA20（取最近 60 个交易日用于窗口计算）"""
    codes_sql = "'" + "','".join(ts_codes) + "'"
    # 从 DB 直接取最近 60 天，用字符串前缀比较（trade_date 是 YYYYMMDD varchar）
    # 计算开始日期：取最新日期往前推约 90 个自然日（保守，覆盖 60 个交易日）
    start_dt  = (pd.Timestamp(latest_price_date) - pd.Timedelta(days=90)).strftime('%Y%m%d')
    df = conn.execute(f"""
        WITH p AS (
            SELECT ts_code, trade_date, close, pct_chg
            FROM daily_price
            WHERE ts_code IN ({codes_sql})
              AND trade_date >= '{start_dt}'
              AND trade_date <= '{latest_price_date}'
            ORDER BY ts_code, trade_date
        )
        SELECT
            ts_code,
            trade_date,
            close,
            pct_chg,
            AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)  AS ma5,
            AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20
        FROM p
    """).fetchdf()
    df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')
    # 取每只股票最新一行
    latest = df.sort_values('trade_date').groupby('ts_code').last().reset_index()
    return latest.set_index('ts_code')


def section_cs_model(target_date: Optional[str] = None, top_n: int = 20):
    print(f"\n{DIVIDER}")
    print("  [3] 截面选股模型 — 最新截面日 Top 股票")
    print(DIVIDER)

    if not CS_PRED_FILE.exists():
        print(f"  [!] 未找到截面选股预测: {CS_PRED_FILE}")
        print("      请先运行: python xgboost_cross_section.py")
        return None, None

    cs = pd.read_csv(CS_PRED_FILE, dtype={'trade_date': str})
    cs['trade_date'] = cs['trade_date'].str.replace('-', '')

    all_dates = sorted(cs['trade_date'].unique())
    if target_date:
        # 找最近的截面日
        avail = [d for d in all_dates if d <= target_date]
        if not avail:
            print(f"  [!] {target_date} 前无截面预测数据")
            return None, None
        cs_date = avail[-1]
    else:
        cs_date = all_dates[-1]

    print(f"\n  截面预测最新日期: {cs_date}  (总截面数: {len(all_dates)})")
    print(f"  预测文件覆盖范围: {all_dates[0]} ~ {all_dates[-1]}")

    day_cs = cs[cs['trade_date'] == cs_date].copy()
    print(f"  本截面股票数: {len(day_cs):,}")
    print(f"  pred 分布:  min={day_cs['pred'].min():.4f}  "
          f"median={day_cs['pred'].median():.4f}  "
          f"max={day_cs['pred'].max():.4f}")

    # 分位数分布
    for q, label in [(0.9, 'Top10%'), (0.75, 'Top25%'), (0.5, '中位数'), (0.25, 'Bot25%')]:
        print(f"    {label}: pred={day_cs['pred'].quantile(q):.4f}")

    # 合规过滤
    with get_conn() as conn:
        eligible = _load_eligible_today(conn, cs_date)
        meta     = _load_stock_meta(conn)

        # 取最新价格日期（DB中的最新）
        latest_price_date = conn.execute(
            "SELECT MAX(trade_date) FROM daily_price WHERE ts_code NOT LIKE '8%'"
        ).fetchone()[0]
        latest_price_date = str(latest_price_date).replace('-', '')

    day_cs_elig = day_cs[day_cs['ts_code'].isin(eligible)].copy()
    print(f"\n  合规过滤后: {len(day_cs_elig):,} 只  (合规率 {len(day_cs_elig)/len(day_cs)*100:.1f}%)")

    day_cs_elig = day_cs_elig.sort_values('pred', ascending=False).reset_index(drop=True)

    # 获取 top_n 股票价格
    top_codes = day_cs_elig['ts_code'].head(top_n).tolist()
    with get_conn() as conn:
        prices = _load_latest_prices(conn, top_codes, latest_price_date)

    print(f"\n  Top-{top_n} 截面预测（价格更新至 {latest_price_date}）:")
    print(f"  {'#':>3}  {'代码':12}  {'名称':10}  {'行业':10}  "
          f"{'pred':>8}  {'分位':>6}  {'当前价':>8}  {'1日涨跌':>8}  {'MA5/MA20':>12}  {'Decile':>7}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*10}  {'-'*10}  "
          f"{'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*7}")

    for i, (_, row) in enumerate(day_cs_elig.head(top_n).iterrows(), 1):
        ts   = row['ts_code']
        pred = row['pred']
        decile = row.get('decile', np.nan)
        pct  = (day_cs['pred'] < pred).mean() * 100  # 全截面百分位

        name     = meta.loc[ts, 'name']     if ts in meta.index else '—'
        industry = meta.loc[ts, 'industry'] if ts in meta.index else '—'
        if isinstance(industry, float):
            industry = '—'
        name     = str(name)[:8]
        industry = str(industry)[:8]

        if ts in prices.index:
            pr = prices.loc[ts]
            close  = pr['close']
            chg    = pr['pct_chg'] / 100 if pd.notna(pr['pct_chg']) else np.nan
            ma5    = pr['ma5']
            ma20   = pr['ma20']
            ma_sym = '▲' if (pd.notna(ma5) and pd.notna(ma20) and ma5 > ma20) else '▼'
            price_str = f"{close:8.2f}"
            chg_str   = fmt_pct(chg)
            ma_str    = f"{ma_sym} {ma5/ma20:.4f}" if (pd.notna(ma5) and pd.notna(ma20)) else '  N/A  '
        else:
            price_str = '    N/A '
            chg_str   = '   N/A  '
            ma_str    = '    N/A     '

        print(f"  {i:>3}  {ts:12}  {name:10}  {industry:10}  "
              f"{pred:8.4f}  {pct:5.1f}%  {price_str}  {chg_str:>8}  {ma_str:>12}  "
              f"{'N/A' if pd.isna(decile) else int(decile):>7}")

    return day_cs_elig, cs_date


# ══════════════════════════════════════════════════════════════════════
# [4] 联合策略建议
# ══════════════════════════════════════════════════════════════════════
def section_recommendation(raw_slots: int, consec_bull: bool,
                            cs_df: Optional[pd.DataFrame],
                            cs_date: Optional[str]):
    print(f"\n{DIVIDER}")
    print("  [4] 联合策略当前建议")
    print(DIVIDER)

    # 判断实际 slots
    if raw_slots == 0:
        effective_slots = 0
        slot_reason = "熊市（CSI300 < MA20）→ 停止开新仓"
    elif not consec_bull:
        effective_slots = 0
        slot_reason = (f"bull 信号存在，但不足 {SLOT_CONFIRM_DAYS} 天连续确认"
                       f"→ 暂不开新仓（SLOT_CONFIRM_DAYS={SLOT_CONFIRM_DAYS}）")
    else:
        # raw_slots 来自 MA 状态（0/10/20），按比例映射到 MAX_SLOTS
        if raw_slots == 10:
            effective_slots = MAX_SLOTS // 2
        else:
            effective_slots = MAX_SLOTS
        slot_reason = f"MA 状态正常 + 连续确认 → {effective_slots} 槽 (MAX_SLOTS={MAX_SLOTS})"

    print(f"\n  市场状态 slots (MA规则): {raw_slots}")
    print(f"  SLOT_CONFIRM_DAYS 确认: {'通过 ✓' if consec_bull else '未通过 ✗'}")
    print(f"  ▶ 有效 slots: {effective_slots}")
    print(f"  ▶ 原因: {slot_reason}")

    if cs_df is None or cs_df.empty:
        print("\n  [!] 无截面预测数据，无法生成持仓建议")
        return

    if effective_slots == 0:
        print(f"\n  ▶ 建议: 不开新仓，现有持仓由风控自然退出（MA死叉/止损）")
        if cs_df is not None:
            print(f"  （截面选股数据日期: {cs_date}，预备用于市场转牛时）")
        return

    print(f"\n  基于截面选股 ({cs_date}) Top-{effective_slots} 建议持仓:")
    print(f"\n  {'#':>3}  {'代码':12}  {'名称':10}  {'pred':>8}  {'说明':}")

    with get_conn() as conn:
        meta = _load_stock_meta(conn)
        latest_price_date = conn.execute(
            "SELECT MAX(trade_date) FROM daily_price WHERE ts_code NOT LIKE '8%'"
        ).fetchone()[0]
        latest_price_date = str(latest_price_date).replace('-', '')
        prices = _load_latest_prices(conn, cs_df['ts_code'].head(effective_slots).tolist(),
                                     latest_price_date)

    for i, (_, row) in enumerate(cs_df.head(effective_slots).iterrows(), 1):
        ts   = row['ts_code']
        pred = row['pred']
        name = meta.loc[ts, 'name'] if ts in meta.index else '—'
        name = str(name)[:10]

        notes = []
        if ts in prices.index:
            pr   = prices.loc[ts]
            ma5  = pr['ma5']
            ma20 = pr['ma20']
            if pd.notna(ma5) and pd.notna(ma20):
                if ma5 > ma20:
                    notes.append('MA5>MA20 ✓')
                else:
                    notes.append('MA5<MA20 ✗ (注意死叉风险)')
            chg = pr['pct_chg']
            if pd.notna(chg):
                notes.append(f"最新涨跌 {fmt_pct(chg/100)}")
        note_str = '  '.join(notes)
        print(f"  {i:>3}  {ts:12}  {name:10}  {pred:8.4f}  {note_str}")

    print(f"\n  ⚠  注意: 截面预测基于 {cs_date} 数据，如与当前日期差距 >10 交易日，")
    print(f"     建议重新运行 xgboost_cross_section.py 以获取最新预测。")


# ══════════════════════════════════════════════════════════════════════
# [5] 最近成交记录
# ══════════════════════════════════════════════════════════════════════
def section_recent_trades(n: int = 15):
    print(f"\n{DIVIDER}")
    print("  [5] 最近成交记录 (index_ma_combined_strategy 回测)")
    print(DIVIDER)

    if not TRADES_FILE.exists():
        print(f"  [!] 未找到交易记录: {TRADES_FILE}")
        return

    trades = pd.read_csv(TRADES_FILE, dtype={'date': str, 'ts_code': str})
    # 过滤掉调仓汇总行
    real_trades = trades[trades['ts_code'] != 'REBAL'].copy()
    rebal_rows  = trades[trades['ts_code'] == 'REBAL'].copy()

    print(f"\n  总交易记录: {len(real_trades)} 笔  "
          f"调仓日: {len(rebal_rows)} 次")

    last_rebal = rebal_rows.iloc[-1] if len(rebal_rows) > 0 else None
    if last_rebal is not None:
        print(f"  最近调仓日: {last_rebal['date']}  "
              f"slots={int(last_rebal['price'])}  "
              f"持仓数={int(last_rebal['cash'])}")

    print(f"\n  最近 {n} 笔个股交易:")
    print(f"  {'日期':10}  {'代码':12}  {'操作':12}  {'价格':>10}  {'现金余额':>14}")

    with get_conn() as conn:
        meta = _load_stock_meta(conn)

    for _, row in real_trades.tail(n).iterrows():
        ts   = row['ts_code']
        name = meta.loc[ts, 'name'] if ts in meta.index else '—'
        name = str(name)[:6]
        print(f"  {row['date']}  {ts:12}  ({name}) {row['action']:10}  "
              f"{row['price']:>10.2f}  {row['cash']:>14,.2f}")

    # 统计（注：买入在 REBAL 调仓汇总行执行，不单独记录）
    stop_cnt = (real_trades['action'] == 'stop_sell').sum()
    ma_cnt   = (real_trades['action'] == 'ma_sell').sum()
    other_cnt = len(real_trades) - stop_cnt - ma_cnt
    print(f"\n  止损卖出: {stop_cnt} 笔  MA死叉卖出: {ma_cnt} 笔  其他: {other_cnt} 笔")
    print(f"  买入: 在调仓日（REBAL）执行，见调仓汇总行 slots/持仓数")


# ══════════════════════════════════════════════════════════════════════
# [6] 数据新鲜度
# ══════════════════════════════════════════════════════════════════════
def section_freshness():
    print(f"\n{DIVIDER}")
    print("  [6] 数据新鲜度")
    print(DIVIDER)

    with get_conn() as conn:
        db_price_latest = conn.execute(
            "SELECT MAX(trade_date) FROM daily_price WHERE ts_code NOT LIKE '8%'"
        ).fetchone()[0]
        db_index_latest = conn.execute(
            "SELECT MAX(trade_date) FROM index_daily"
        ).fetchone()[0]

    db_price_latest = str(db_price_latest).replace('-', '')
    db_index_latest = str(db_index_latest).replace('-', '')

    rows = [
        ('DB daily_price',  db_price_latest,  '个股价格'),
        ('DB index_daily',  db_index_latest,  'CSI300 MA状态'),
    ]

    if INDEX_TIMING_FILE.exists():
        t = pd.read_csv(INDEX_TIMING_FILE, dtype={'trade_date': str})
        t['trade_date'] = t['trade_date'].str.replace('-', '')
        rows.append(('时序模型预测', t['trade_date'].max(), 'pred_prob/slots'))

    if CS_PRED_FILE.exists():
        c = pd.read_csv(CS_PRED_FILE, dtype={'trade_date': str})
        c['trade_date'] = c['trade_date'].str.replace('-', '')
        rows.append(('截面选股预测', c['trade_date'].max(), 'CS pred score'))

    print(f"\n  {'数据源':16}  {'最新日期':12}  {'用途':20}  {'状态'}")
    for name, date, use in rows:
        delta_str = ''
        try:
            d1 = pd.Timestamp(db_price_latest)
            d2 = pd.Timestamp(date)
            delta = (d1 - d2).days
            delta_str = f"落后 {delta} 天" if delta > 0 else "最新"
        except Exception:
            delta_str = '—'
        print(f"  {name:16}  {date:12}  {use:20}  {delta_str}")

    print(f"""
  如需刷新预测（约需 5~30 分钟）:
    # 刷新指数择时（~5分钟）
    python index_timing_model.py --label_type ma60_state --no_wfo

    # 刷新截面选股（~25分钟）
    python xgboost_cross_section.py

    # 刷新完成后重新运行本脚本
    python daily_inference.py
""")


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='每日 Inference — 联合量化策略')
    parser.add_argument('--top_n', type=int, default=20,
                        help='截面选股展示前 N 名（默认 20）')
    parser.add_argument('--date', type=str, default=None,
                        help='指定截面日期 YYYYMMDD（默认最新）')
    args = parser.parse_args()

    print(f"\n{'═'*72}")
    print("  每日 Inference — 指数择时 + 截面选股联合策略")
    print(f"{'═'*72}")
    import datetime
    print(f"  运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    section_db_status()

    raw_slots, consec_bull = section_timing(display_days=20)

    cs_df, cs_date = section_cs_model(target_date=args.date, top_n=args.top_n)

    section_recommendation(raw_slots, consec_bull, cs_df, cs_date)

    section_recent_trades(n=15)

    section_freshness()

    print(f"\n{DIVIDER}\n")


if __name__ == '__main__':
    main()
