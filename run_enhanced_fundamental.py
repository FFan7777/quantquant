#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强型基本面策略回测 v2（时序版，无未来信息）

买入条件（全部满足）：
  - 市值 30亿–500亿
  - F-Score >= 6（9维 Piotroski 同比，从原始报表计算）
  - ROE > 10%
  - MA5 > MA20（趋势向上）
  - RSI(14) < 60（买入条件，持有不检查）
  - 量比 > 1.5（买入条件，持有不检查）
  - PE 横截面百分位 < 80%（买入条件，持有不检查）

持有条件（任一违反则清仓）：
  - F-Score >= 6  AND  ROE > 10%
  - MA5 > MA20
  - 持有期最高点回撤 < 7%
  - 价格 > 成本价 × 90%

评分系统：
  - 基本面 40% + 技术面 30% + 资金面 30%
  - 每次买入/调整取综合评分 top20

仓位控制：
  - 每只股票最多 20%；不足 5 只时剩余持现金
  - 上证指数 < MA20 时总持仓上限 50%
"""

import pandas as pd
import duckdb
import time
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

from data_collect.config import config
from backtesting.performance_metrics import PerformanceMetrics
from backtesting.vectorized_backtest_engine import VectorizedBacktestEngine
from policies.enhanced_fundamental_strategy import EnhancedFundamentalStrategy


def run_enhanced_backtest():
    print("\n" + "=" * 60)
    print("增强型基本面策略回测（时序版，无未来信息）")
    print("=" * 60)
    print(f"回测期间: 2021-02-23 至 2026-02-23（5年）")

    total_start = time.time()

    # ----------------------------------------------------------------
    # 1. 加载行情数据
    # ----------------------------------------------------------------
    print("\n[1/6] 加载行情数据...")
    t0 = time.time()

    conn = duckdb.connect(config.db_path, read_only=True)
    data = conn.execute("""
        SELECT ts_code, trade_date, close, vol
        FROM daily_price
        WHERE trade_date >= '20210223' AND trade_date <= '20260223'
          AND close >= 5.0
        ORDER BY trade_date, ts_code
    """).fetchdf()
    sse_index = conn.execute("""
        SELECT trade_date, close
        FROM index_daily
        WHERE ts_code = '000001.SH'
          AND trade_date >= '20210223' AND trade_date <= '20260223'
        ORDER BY trade_date
    """).fetchdf()
    conn.close()

    load_time = time.time() - t0
    print(f"✓ 行情加载完成 (耗时: {load_time:.2f}秒)")
    print(f"  总记录数: {len(data):,}")
    print(f"  股票数量: {data['ts_code'].nunique()}")
    print(f"  交易日数: {data['trade_date'].nunique()}")
    print(f"  上证指数: {len(sse_index)} 个交易日")

    # ----------------------------------------------------------------
    # 2. 创建策略
    # ----------------------------------------------------------------
    print("\n[2/6] 创建增强型基本面策略...")
    strategy = EnhancedFundamentalStrategy(
        top_n=20,
        min_fscore=6,
        max_pe_percentile=80.0,
        min_roe=10.0,
        ma_short=5,
        ma_long=20,
        rsi_period=14,
        rsi_threshold=60.0,
        vol_ratio_period=5,
        vol_ratio_threshold=1.5,
        stop_loss_pct=0.07,
        cost_stop_loss_pct=0.10,
        min_mktcap=30.0,
        max_mktcap=500.0,
        name="Enhanced_Fundamental_v3",
    )

    strategy.set_market_index(sse_index, max_position_pct=0.5, ma_period=20, ma_long_period=60)

    # ----------------------------------------------------------------
    # 3. 加载时序基本面数据（无未来信息）
    # ----------------------------------------------------------------
    print("\n[3/6] 加载时序基本面数据...")
    t_fund = time.time()
    strategy.load_time_series_fundamentals(config.db_path)
    fund_time = time.time() - t_fund

    info = strategy.get_strategy_info()
    print("\n买入条件:")
    for c in info.get('buy_conditions', []):
        print(f"  ✓ {c}")
    print("\n持有条件（违反则清仓）:")
    for c in info.get('hold_conditions', []):
        print(f"  ✗ {c}")
    print(f"\n评分: {info.get('scoring', '')}")
    print(f"仓位控制: {info.get('position_control', '')}")

    # ----------------------------------------------------------------
    # 4. 创建回测引擎
    # ----------------------------------------------------------------
    print("\n[4/6] 创建回测引擎...")
    engine = VectorizedBacktestEngine(
        initial_capital=1_000_000.0,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        transfer_fee_rate=0.00002,
        slippage_rate=0.0001,
        min_commission=5.0,
        min_trade_value=2000.0,   # 跳过 <2000 元的微小漂移修正交易
    )

    # ----------------------------------------------------------------
    # 5. 运行回测
    # ----------------------------------------------------------------
    print("\n[5/6] 运行回测...")
    t0 = time.time()
    equity_curve = engine.run(strategy, data)
    backtest_time = time.time() - t0
    print(f"✓ 回测完成 (耗时: {backtest_time:.2f}秒)")

    # ----------------------------------------------------------------
    # 6. 性能指标
    # ----------------------------------------------------------------
    print("\n[6/6] 计算性能指标...")
    metrics_calc = PerformanceMetrics(risk_free_rate=0.03)
    metrics = metrics_calc.calculate_all_metrics(equity_curve)
    metrics_calc.print_metrics(metrics)

    # ----------------------------------------------------------------
    # 交易统计
    # ----------------------------------------------------------------
    trades = engine.get_trades()
    if len(trades) > 0:
        print("\n" + "=" * 60)
        print("交易统计")
        print("=" * 60)

        buys  = trades[trades['action'] == 'buy']
        sells = trades[trades['action'] == 'sell']
        print(f"\n总交易次数: {len(trades)}")
        print(f"  买入: {len(buys)}")
        print(f"  卖出: {len(sells)}")

        rebalance_dates = trades['date'].unique()
        print(f"触发调整日数: {len(rebalance_dates)}")
        print(f"平均每次调整交易数: {len(trades) / len(rebalance_dates):.1f}")

        trades.to_csv(
            'output/csv/enhanced_fundamental_trades.csv',
            index=False, encoding='utf-8-sig'
        )
        print(f"\n✓ 交易记录已保存: output/csv/enhanced_fundamental_trades.csv")

    equity_curve.to_csv(
        'output/csv/enhanced_fundamental_equity.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"✓ 权益曲线已保存: output/csv/enhanced_fundamental_equity.csv")

    # ----------------------------------------------------------------
    # 年度收益
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("年度收益分析")
    print("=" * 60)

    equity_curve['year'] = equity_curve['date'].astype(str).str[:4]
    yearly_rows = []
    for year in sorted(equity_curve['year'].unique()):
        yr = equity_curve[equity_curve['year'] == year]
        if len(yr) == 0:
            continue
        start_val = yr.iloc[0]['total_value']
        end_val   = yr.iloc[-1]['total_value']
        ret = (end_val - start_val) / start_val
        yearly_rows.append({
            '年份':   year,
            '收益率': f"{ret * 100:.2f}%",
            '期初资产': f"{start_val:,.0f}",
            '期末资产': f"{end_val:,.0f}",
        })
    print("\n" + pd.DataFrame(yearly_rows).to_string(index=False))

    # ----------------------------------------------------------------
    # 耗时汇总
    # ----------------------------------------------------------------
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("耗时汇总")
    print("=" * 60)
    print(f"总耗时:         {total_time:.2f}秒")
    print(f"  行情加载:     {load_time:.2f}秒  ({load_time / total_time * 100:.1f}%)")
    print(f"  基本面加载:   {fund_time:.2f}秒  ({fund_time / total_time * 100:.1f}%)")
    print(f"  回测计算:     {backtest_time:.2f}秒  ({backtest_time / total_time * 100:.1f}%)")
    trading_days = data['trade_date'].nunique()
    print(f"\n平均每交易日: {backtest_time / trading_days * 1000:.2f}毫秒")

    print("\n" + "=" * 60)
    print("回测完成")
    print("=" * 60)

    return equity_curve, metrics, trades


def main():
    print("\n" + "=" * 60)
    print("增强型基本面策略完整示例")
    print("=" * 60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        run_enhanced_backtest()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
