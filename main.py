#!/usr/bin/env python
"""
A股数据收集主程序

使用方法:
    python main.py init                   # 初始化数据库
    python main.py collect-all            # 全量收集所有数据（含所有表）
    python main.py update                 # 增量更新数据（日常使用）
    python main.py collect-stock          # 只收集股票列表
    python main.py collect-daily          # 只收集日线数据
    python main.py collect-financial      # 只收集财务数据（三张报表）
    python main.py collect-moneyflow-fast # 并发全量收集资金流向（含断点续传）
    python main.py collect-report-rc      # 收集券商一致预期数据
    python main.py stats                  # 显示数据统计
    python main.py refetch-missing        # 重新获取遗漏数据
    python main.py collect-holder         # 收集股东户数（按季度）
"""

import argparse
import sys
from datetime import datetime

from data_collect import DataCollector, DatabaseManager, MissingDataRefetcher


def init_database():
    """初始化数据库"""
    print("初始化数据库...")
    db = DatabaseManager()
    db.init_tables()
    print("数据库初始化完成")


def collect_all(start_date: str = None, end_date: str = None):
    """全量收集所有数据"""
    collector = DataCollector()
    collector.collect_all(start_date=start_date, end_date=end_date)


def collect_stock():
    """收集股票列表"""
    collector = DataCollector()
    collector.collect_stock_basic()


def collect_daily(start_date: str = None, end_date: str = None, incremental: bool = False):
    """收集日线数据"""
    collector = DataCollector()
    collector.collect_daily_price(start_date=start_date, end_date=end_date, incremental=incremental)


def collect_financial(start_date: str = None, end_date: str = None):
    """收集财务数据"""
    collector = DataCollector()
    collector.collect_financial_data(start_date=start_date, end_date=end_date)


def incremental_update():
    """增量更新"""
    collector = DataCollector()
    collector.incremental_update()


def show_stats():
    """显示数据统计"""
    db = DatabaseManager()
    db.connect()

    print("\n" + "=" * 80)
    print("数据库统计信息")
    print("=" * 80)

    # 股票数量
    stock_count = db.query("SELECT COUNT(*) as count FROM stock_basic").iloc[0]['count']
    print(f"股票数量: {stock_count}")

    # 日线数据
    daily_stats = db.query("""
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT ts_code) as stock_count,
            MIN(trade_date) as min_date,
            MAX(trade_date) as max_date
        FROM daily_price
    """)
    if not daily_stats.empty and daily_stats.iloc[0]['total_records'] > 0:
        stats = daily_stats.iloc[0]
        print(f"\n日线数据:")
        print(f"  总记录数: {stats['total_records']:,}")
        print(f"  股票数: {stats['stock_count']}")
        print(f"  日期范围: {stats['min_date']} - {stats['max_date']}")

    # 财务数据
    for table_name, display_name in [
        ('balance_sheet', '资产负债表'),
        ('income_statement', '利润表'),
        ('cash_flow', '现金流量表')
    ]:
        count = db.query(f"SELECT COUNT(*) as count FROM {table_name}").iloc[0]['count']
        print(f"\n{display_name}: {count:,} 条记录")

    # 其他数据表
    for table_name, display_name in [
        ('daily_basic', '每日指标'),
        ('fina_indicator', '财务指标'),
        ('moneyflow', '资金流向'),
        ('index_daily', '指数日线'),
        ('dividend', '分红送股'),
        ('report_rc', '券商预测')
    ]:
        count = db.query(f"SELECT COUNT(*) as count FROM {table_name}").iloc[0]['count']
        print(f"\n{display_name}: {count:,} 条记录")

    # 最近更新日志
    print("\n" + "=" * 80)
    print("最近更新日志 (最近5条)")
    print("=" * 80)
    logs = db.get_update_logs(limit=5)
    if not logs.empty:
        for _, log in logs.iterrows():
            print(f"\n表名: {log['table_name']}")
            print(f"  类型: {log['update_type']}")
            print(f"  日期: {log['start_date']} - {log['end_date']}")
            print(f"  记录数: {log['records_count']}")
            print(f"  状态: {log['status']}")
            print(f"  时间: {log['created_at']}")
            if log['error_message']:
                print(f"  错误: {log['error_message']}")

    db.close()


def collect_moneyflow_fast(start_date=None, end_date=None,
                           resume: bool = True, workers: int = 5):
    """并发全量收集资金流向数据（含断点续传）"""
    collector = DataCollector()
    collector.collect_moneyflow_fast(start_date=start_date, end_date=end_date,
                                     resume=resume, workers=workers)


def collect_report_rc(start_date=None, end_date=None, incremental: bool = False):
    """收集券商一致预期数据"""
    collector = DataCollector()
    collector.collect_report_rc(start_date=start_date, end_date=end_date,
                                incremental=incremental)


def refetch_missing():
    """重新获取遗漏数据"""
    refetcher = MissingDataRefetcher()
    try:
        refetcher.refetch_all()
    finally:
        refetcher.close()


def collect_holder(start_year: int = 2016, end_year: int = None):
    """收集股东户数（stk_holdernumber，按季度）"""
    collector = DataCollector()
    collector.collect_holder_data(start_year=start_year, end_year=end_year)


def main():
    parser = argparse.ArgumentParser(description='A股数据收集工具')
    parser.add_argument('command', choices=[
        'init', 'collect-all', 'update', 'collect-stock',
        'collect-daily', 'collect-financial',
        'collect-moneyflow-fast', 'collect-report-rc',
        'stats', 'refetch-missing', 'collect-holder'
    ], help='执行的命令')
    parser.add_argument('--start-date', help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', help='结束日期 (YYYYMMDD)')
    parser.add_argument('--incremental', action='store_true', help='增量更新模式')
    parser.add_argument('--no-resume', action='store_true', help='collect-moneyflow-fast: 不使用断点续传')
    parser.add_argument('--workers', type=int, default=5, help='collect-moneyflow-fast: 并发线程数 (默认: 5)')
    parser.add_argument('--start-year', type=int, default=2016, help='collect-holder: 开始年份 (默认: 2016)')
    parser.add_argument('--end-year',   type=int, default=None,  help='collect-holder: 结束年份 (默认: 当年)')

    args = parser.parse_args()

    try:
        if args.command == 'init':
            init_database()

        elif args.command == 'collect-all':
            collect_all(start_date=args.start_date, end_date=args.end_date)

        elif args.command == 'update':
            incremental_update()

        elif args.command == 'collect-stock':
            collect_stock()

        elif args.command == 'collect-daily':
            collect_daily(
                start_date=args.start_date,
                end_date=args.end_date,
                incremental=args.incremental
            )

        elif args.command == 'collect-financial':
            collect_financial(start_date=args.start_date, end_date=args.end_date)

        elif args.command == 'collect-moneyflow-fast':
            collect_moneyflow_fast(
                start_date=args.start_date,
                end_date=args.end_date,
                resume=not args.no_resume,
                workers=args.workers
            )

        elif args.command == 'collect-report-rc':
            collect_report_rc(
                start_date=args.start_date,
                end_date=args.end_date,
                incremental=args.incremental
            )

        elif args.command == 'stats':
            show_stats()

        elif args.command == 'refetch-missing':
            refetch_missing()

        elif args.command == 'collect-holder':
            collect_holder(start_year=args.start_year, end_year=args.end_year)

    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
