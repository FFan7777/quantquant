"""
遗漏数据检查与重新获取
"""
from datetime import datetime
from typing import List, Dict
import pandas as pd

from .config import config
from .database import DatabaseManager
from .collector import DataCollector


class MissingDataRefetcher:
    """遗漏数据重新获取器"""

    def __init__(self):
        self.db = DatabaseManager()
        self.collector = DataCollector()
        self.db.connect()

    def get_all_stocks(self) -> List[str]:
        """获取所有股票代码"""
        stock_df = self.db.get_stock_list()
        if stock_df.empty:
            print("警告: 股票列表为空")
            return []
        return stock_df['ts_code'].tolist()

    def check_daily_price_missing(self, start_date: str, end_date: str) -> List[str]:
        """检查日线数据中缺失的股票"""
        print("\n检查日线数据完整性...")

        all_stocks = self.get_all_stocks()
        if not all_stocks:
            return []

        existing_df = self.db.query(f"""
            SELECT DISTINCT ts_code,
                   COUNT(*) as record_count,
                   MIN(trade_date) as min_date,
                   MAX(trade_date) as max_date
            FROM daily_price
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            GROUP BY ts_code
        """)

        if existing_df.empty:
            print(f"  所有 {len(all_stocks)} 只股票都缺失日线数据")
            return all_stocks

        existing_stocks = set(existing_df['ts_code'].tolist())
        missing_stocks = [code for code in all_stocks if code not in existing_stocks]
        incomplete_stocks = [row['ts_code'] for _, row in existing_df.iterrows() if row['record_count'] < 10]
        all_missing = list(set(missing_stocks + incomplete_stocks))

        print(f"  完全缺失: {len(missing_stocks)} 只股票")
        print(f"  数据不完整: {len(incomplete_stocks)} 只股票")
        print(f"  需要重新获取: {len(all_missing)} 只股票")
        return all_missing

    def check_financial_missing(self, start_date: str, end_date: str) -> Dict[str, List[str]]:
        """检查财务数据中缺失的股票"""
        print("\n检查财务数据完整性...")

        all_stocks = self.get_all_stocks()
        if not all_stocks:
            return {}

        tables = {
            'balance_sheet': '资产负债表',
            'income_statement': '利润表',
            'cash_flow': '现金流量表'
        }
        missing_dict = {}

        for table_name, display_name in tables.items():
            existing_df = self.db.query(f"""
                SELECT DISTINCT ts_code, COUNT(*) as record_count
                FROM {table_name}
                WHERE end_date >= '{start_date}' AND end_date <= '{end_date}'
                GROUP BY ts_code
            """)

            if existing_df.empty:
                missing_dict[table_name] = all_stocks
                print(f"  {display_name}: 所有 {len(all_stocks)} 只股票都缺失数据")
            else:
                existing_stocks = set(existing_df['ts_code'].tolist())
                missing_stocks = [code for code in all_stocks if code not in existing_stocks]
                incomplete_stocks = [row['ts_code'] for _, row in existing_df.iterrows() if row['record_count'] < 2]
                all_missing = list(set(missing_stocks + incomplete_stocks))
                missing_dict[table_name] = all_missing
                print(f"  {display_name}: 需要重新获取 {len(all_missing)} 只股票")

        return missing_dict

    def check_daily_basic_missing(self, start_date: str, end_date: str) -> List[str]:
        """检查每日指标数据缺失"""
        print("\n检查每日指标数据完整性...")

        all_stocks = self.get_all_stocks()
        if not all_stocks:
            return []

        existing_df = self.db.query(f"""
            SELECT DISTINCT ts_code, COUNT(*) as record_count
            FROM daily_basic
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            GROUP BY ts_code
        """)

        if existing_df.empty:
            print(f"  所有 {len(all_stocks)} 只股票都缺失数据")
            return all_stocks

        existing_stocks = set(existing_df['ts_code'].tolist())
        missing_stocks = [code for code in all_stocks if code not in existing_stocks]
        incomplete_stocks = [row['ts_code'] for _, row in existing_df.iterrows() if row['record_count'] < 10]
        all_missing = list(set(missing_stocks + incomplete_stocks))
        print(f"  需要重新获取: {len(all_missing)} 只股票")
        return all_missing

    def check_fina_indicator_missing(self, start_date: str, end_date: str) -> List[str]:
        """检查财务指标数据缺失"""
        print("\n检查财务指标数据完整性...")

        all_stocks = self.get_all_stocks()
        if not all_stocks:
            return []

        existing_df = self.db.query(f"""
            SELECT DISTINCT ts_code, COUNT(*) as record_count
            FROM fina_indicator
            WHERE end_date >= '{start_date}' AND end_date <= '{end_date}'
            GROUP BY ts_code
        """)

        if existing_df.empty:
            print(f"  所有 {len(all_stocks)} 只股票都缺失数据")
            return all_stocks

        existing_stocks = set(existing_df['ts_code'].tolist())
        missing_stocks = [code for code in all_stocks if code not in existing_stocks]
        incomplete_stocks = [row['ts_code'] for _, row in existing_df.iterrows() if row['record_count'] < 2]
        all_missing = list(set(missing_stocks + incomplete_stocks))
        print(f"  需要重新获取: {len(all_missing)} 只股票")
        return all_missing

    def check_moneyflow_missing(self, start_date: str, end_date: str) -> List[str]:
        """检查资金流向数据缺失"""
        print("\n检查资金流向数据完整性...")

        all_stocks = self.get_all_stocks()
        if not all_stocks:
            return []

        existing_df = self.db.query(f"""
            SELECT DISTINCT ts_code, COUNT(*) as record_count
            FROM moneyflow
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            GROUP BY ts_code
        """)

        if existing_df.empty:
            print(f"  所有 {len(all_stocks)} 只股票都缺失数据")
            return all_stocks

        existing_stocks = set(existing_df['ts_code'].tolist())
        missing_stocks = [code for code in all_stocks if code not in existing_stocks]
        incomplete_stocks = [row['ts_code'] for _, row in existing_df.iterrows() if row['record_count'] < 10]
        all_missing = list(set(missing_stocks + incomplete_stocks))
        print(f"  需要重新获取: {len(all_missing)} 只股票")
        return all_missing

    def check_dividend_missing(self) -> List[str]:
        """检查分红数据缺失"""
        print("\n检查分红数据完整性...")

        all_stocks = self.get_all_stocks()
        if not all_stocks:
            return []

        existing_df = self.db.query("SELECT DISTINCT ts_code FROM dividend")
        if existing_df.empty:
            print(f"  所有 {len(all_stocks)} 只股票都缺失数据")
            return all_stocks

        existing_stocks = set(existing_df['ts_code'].tolist())
        missing_stocks = [code for code in all_stocks if code not in existing_stocks]
        print(f"  需要重新获取: {len(missing_stocks)} 只股票")
        return missing_stocks

    def refetch_daily_price(self, missing_stocks: List[str], start_date: str, end_date: str):
        """重新获取日线数据"""
        if not missing_stocks:
            print("\n日线数据完整，无需重新获取")
            return
        print(f"\n开始重新获取 {len(missing_stocks)} 只股票的日线数据...")
        self.collector.collect_daily_price(start_date=start_date, end_date=end_date,
                                           ts_codes=missing_stocks, incremental=False)

    def refetch_financial_data(self, missing_dict: Dict[str, List[str]],
                               start_date: str, end_date: str):
        """重新获取财务数据"""
        all_missing = set()
        for stocks in missing_dict.values():
            all_missing.update(stocks)

        if not all_missing:
            print("\n财务数据完整，无需重新获取")
            return

        print(f"\n开始重新获取 {len(all_missing)} 只股票的财务数据...")
        self.collector.collect_financial_data(start_date=start_date, end_date=end_date,
                                              ts_codes=list(all_missing))

    def refetch_daily_basic(self, missing_stocks: List[str], start_date: str, end_date: str):
        """重新获取每日指标数据"""
        if not missing_stocks:
            print("\n每日指标数据完整，无需重新获取")
            return
        print(f"\n开始重新获取 {len(missing_stocks)} 只股票的每日指标数据...")
        self.collector.collect_daily_basic(start_date=start_date, end_date=end_date,
                                           ts_codes=missing_stocks, incremental=False)

    def refetch_fina_indicator(self, missing_stocks: List[str], start_date: str, end_date: str):
        """重新获取财务指标数据"""
        if not missing_stocks:
            print("\n财务指标数据完整，无需重新获取")
            return
        print(f"\n开始重新获取 {len(missing_stocks)} 只股票的财务指标数据...")
        self.collector.collect_fina_indicator(start_date=start_date, end_date=end_date,
                                              ts_codes=missing_stocks)

    def refetch_moneyflow(self, missing_stocks: List[str], start_date: str, end_date: str):
        """重新获取资金流向数据"""
        if not missing_stocks:
            print("\n资金流向数据完整，无需重新获取")
            return
        print(f"\n开始重新获取 {len(missing_stocks)} 只股票的资金流向数据...")
        self.collector.collect_moneyflow(start_date=start_date, end_date=end_date,
                                         ts_codes=missing_stocks, incremental=False)

    def refetch_dividend(self, missing_stocks: List[str]):
        """重新获取分红数据"""
        if not missing_stocks:
            print("\n分红数据完整，无需重新获取")
            return
        print(f"\n开始重新获取 {len(missing_stocks)} 只股票的分红数据...")
        self.collector.collect_dividend(ts_codes=missing_stocks)

    def refetch_all(self, start_date: str = None, end_date: str = None):
        """检查并重新获取所有遗漏数据"""
        print("=" * 80)
        print("开始检查所有表的数据完整性")
        print("=" * 80)

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = config.start_date

        print(f"\n检查日期范围: {start_date} - {end_date}")

        missing_daily = self.check_daily_price_missing(start_date, end_date)
        if missing_daily:
            self.refetch_daily_price(missing_daily, start_date, end_date)

        missing_financial = self.check_financial_missing(start_date, end_date)
        if missing_financial:
            self.refetch_financial_data(missing_financial, start_date, end_date)

        missing_daily_basic = self.check_daily_basic_missing(start_date, end_date)
        if missing_daily_basic:
            self.refetch_daily_basic(missing_daily_basic, start_date, end_date)

        missing_fina = self.check_fina_indicator_missing(start_date, end_date)
        if missing_fina:
            self.refetch_fina_indicator(missing_fina, start_date, end_date)

        missing_moneyflow = self.check_moneyflow_missing(start_date, end_date)
        if missing_moneyflow:
            self.refetch_moneyflow(missing_moneyflow, start_date, end_date)

        missing_dividend = self.check_dividend_missing()
        if missing_dividend:
            self.refetch_dividend(missing_dividend)

        print("\n" + "=" * 80)
        print("数据完整性检查和重新获取完成")
        print("=" * 80)

    def close(self):
        """关闭数据库连接"""
        self.db.close()
