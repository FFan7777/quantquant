"""
数据收集器主模块
"""
import json
import threading
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import config
from .database import DatabaseManager
from .tushare_api import TushareAPI


# ---------------------------------------------------------------------------
# 资金流向并发收集辅助函数（模块级，供 ThreadPoolExecutor worker 使用）
# ---------------------------------------------------------------------------

def _load_moneyflow_progress() -> dict:
    """加载资金流向收集断点进度"""
    progress_file = Path('data/moneyflow_progress.json')
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_stocks': [], 'last_index': 0}


def _save_moneyflow_progress(progress: dict):
    """保存资金流向收集断点进度"""
    progress_file = Path('data/moneyflow_progress.json')
    with open(progress_file, 'w') as f:
        json.dump(progress, f)


def _fetch_stock_moneyflow(ts_code: str, name: str, start_date: str,
                           end_date: str, api: TushareAPI) -> tuple:
    """并发 worker：获取单只股票的资金流向数据"""
    try:
        df = api.get_moneyflow(ts_code, start_date, end_date)
        if df is not None and not df.empty:
            return (ts_code, df, None)
        return (ts_code, None, None)
    except Exception as e:
        return (ts_code, None, str(e))


class DataCollector:
    """数据收集器"""

    def __init__(self):
        """初始化数据收集器"""
        self.api = TushareAPI()
        self.db = DatabaseManager()
        self.db.init_tables()

    def collect_stock_basic(self):
        """收集股票基本信息"""
        print("\n" + "=" * 80)
        print("开始收集股票基本信息...")
        print("=" * 80)

        try:
            # 获取上市股票
            stock_df = self.api.get_stock_basic(list_status='L')

            if config.include_delisted:
                # 获取退市股票
                delisted_df = self.api.get_stock_basic(list_status='D')
                stock_df = pd.concat([stock_df, delisted_df], ignore_index=True)

            print(f"获取到 {len(stock_df)} 只股票")

            # 保存到数据库
            self.db.insert_dataframe('stock_basic', stock_df, mode='replace')

            # 记录日志
            self.db.log_update(
                table_name='stock_basic',
                update_type='full',
                start_date='',
                end_date='',
                records_count=len(stock_df),
                status='success'
            )

            print("股票基本信息收集完成")
            return stock_df

        except Exception as e:
            print(f"收集股票基本信息失败: {e}")
            self.db.log_update(
                table_name='stock_basic',
                update_type='full',
                start_date='',
                end_date='',
                records_count=0,
                status='failed',
                error_message=str(e)
            )
            raise

    def collect_daily_price_single(self, ts_code: str, start_date: str, end_date: str) -> int:
        """
        收集单只股票的日线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            收集的记录数
        """
        try:
            # 获取带前复权的日线数据
            daily_df = self.api.get_daily_with_adj(ts_code, start_date, end_date)

            if daily_df.empty:
                return 0

            # 为每个线程创建独立的数据库连接
            db = DatabaseManager()
            db.insert_dataframe('daily_price', daily_df)
            db.close()

            return len(daily_df)

        except Exception as e:
            print(f"收集 {ts_code} 日线数据失败: {e}")
            return 0

    def collect_daily_price(self, start_date: str = None, end_date: str = None,
                           ts_codes: List[str] = None, incremental: bool = False):
        """
        收集日线数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            ts_codes: 股票代码列表，如果为 None 则收集所有股票
            incremental: 是否增量更新
        """
        print("\n" + "=" * 80)
        print("开始收集日线数据...")
        print("=" * 80)

        # 设置日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            if incremental:
                # 增量更新：从最新日期开始
                latest_date = self.db.get_latest_trade_date()
                if latest_date:
                    start_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
                else:
                    start_date = config.start_date
            else:
                start_date = config.start_date

        print(f"日期范围: {start_date} - {end_date}")

        # 获取股票列表
        if ts_codes is None:
            stock_df = self.db.get_stock_list()
            if stock_df.empty:
                print("股票列表为空，请先收集股票基本信息")
                return
            ts_codes = stock_df['ts_code'].tolist()

        print(f"待收集股票数: {len(ts_codes)}")

        # 并发收集数据
        total_records = 0
        failed_stocks = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # 提交任务
            future_to_code = {
                executor.submit(self.collect_daily_price_single, code, start_date, end_date): code
                for code in ts_codes
            }

            # 使用进度条
            with tqdm(total=len(ts_codes), desc="收集日线数据") as pbar:
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        records = future.result()
                        total_records += records
                    except Exception as e:
                        print(f"\n{code} 收集失败: {e}")
                        failed_stocks.append(code)
                    finally:
                        pbar.update(1)

        # 记录日志
        update_type = 'incremental' if incremental else 'full'
        status = 'success' if not failed_stocks else 'partial'
        error_msg = f"失败股票: {','.join(failed_stocks[:10])}" if failed_stocks else None

        self.db.log_update(
            table_name='daily_price',
            update_type=update_type,
            start_date=start_date,
            end_date=end_date,
            records_count=total_records,
            status=status,
            error_message=error_msg
        )

        print(f"\n日线数据收集完成: 总记录数 {total_records}")
        if failed_stocks:
            print(f"失败股票数: {len(failed_stocks)}")

    def collect_financial_single(self, ts_code: str, start_date: str, end_date: str) -> dict:
        """
        收集单只股票的财务数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            收集的记录数字典
        """
        result = {'balance_sheet': 0, 'income_statement': 0, 'cash_flow': 0}

        try:
            # 为每个线程创建独立的数据库连接
            db = DatabaseManager()

            # 资产负债表
            balance_df = self.api.get_balancesheet(ts_code, start_date, end_date)
            if not balance_df.empty:
                db.insert_dataframe('balance_sheet', balance_df)
                result['balance_sheet'] = len(balance_df)

            # 利润表
            income_df = self.api.get_income(ts_code, start_date, end_date)
            if not income_df.empty:
                db.insert_dataframe('income_statement', income_df)
                result['income_statement'] = len(income_df)

            # 现金流量表
            cashflow_df = self.api.get_cashflow(ts_code, start_date, end_date)
            if not cashflow_df.empty:
                db.insert_dataframe('cash_flow', cashflow_df)
                result['cash_flow'] = len(cashflow_df)

            db.close()

        except Exception as e:
            print(f"收集 {ts_code} 财务数据失败: {e}")

        return result

    def collect_financial_data(self, start_date: str = None, end_date: str = None,
                              ts_codes: List[str] = None):
        """
        收集财务报表数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            ts_codes: 股票代码列表
        """
        print("\n" + "=" * 80)
        print("开始收集财务报表数据...")
        print("=" * 80)

        # 设置日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            start_date = config.start_date

        print(f"日期范围: {start_date} - {end_date}")

        # 获取股票列表
        if ts_codes is None:
            stock_df = self.db.get_stock_list()
            if stock_df.empty:
                print("股票列表为空，请先收集股票基本信息")
                return
            ts_codes = stock_df['ts_code'].tolist()

        print(f"待收集股票数: {len(ts_codes)}")

        # 并发收集数据
        total_records = {'balance_sheet': 0, 'income_statement': 0, 'cash_flow': 0}
        failed_stocks = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_code = {
                executor.submit(self.collect_financial_single, code, start_date, end_date): code
                for code in ts_codes
            }

            with tqdm(total=len(ts_codes), desc="收集财务数据") as pbar:
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        result = future.result()
                        for key in total_records:
                            total_records[key] += result[key]
                    except Exception as e:
                        print(f"\n{code} 收集失败: {e}")
                        failed_stocks.append(code)
                    finally:
                        pbar.update(1)

        # 记录日志
        for table_name, count in total_records.items():
            status = 'success' if not failed_stocks else 'partial'
            error_msg = f"失败股票: {','.join(failed_stocks[:10])}" if failed_stocks else None

            self.db.log_update(
                table_name=table_name,
                update_type='full',
                start_date=start_date,
                end_date=end_date,
                records_count=count,
                status=status,
                error_message=error_msg
            )

        print(f"\n财务数据收集完成:")
        print(f"  资产负债表: {total_records['balance_sheet']} 条")
        print(f"  利润表: {total_records['income_statement']} 条")
        print(f"  现金流量表: {total_records['cash_flow']} 条")
        if failed_stocks:
            print(f"失败股票数: {len(failed_stocks)}")

    def collect_all(self, start_date: str = None, end_date: str = None):
        """
        收集所有数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        print("\n" + "=" * 80)
        print("开始全量数据收集")
        print("=" * 80)

        # 1. 收集股票基本信息
        self.collect_stock_basic()

        # 2. 收集日线数据（前复权）
        self.collect_daily_price(start_date=start_date, end_date=end_date, incremental=False)

        # 3. 收集财务数据（三张报表）
        self.collect_financial_data(start_date=start_date, end_date=end_date)

        # 4. 收集每日指标（市值/PE/换手率等）
        self.collect_daily_basic(start_date=start_date, end_date=end_date)

        # 5. 收集财务指标（ROE/ROA等，按季度计算值）
        self.collect_fina_indicator(start_date=start_date, end_date=end_date)

        # 6. 收集资金流向
        self.collect_moneyflow(start_date=start_date, end_date=end_date)

        # 7. 收集指数数据（日线 + 基础信息）
        self.collect_index_data(start_date=start_date, end_date=end_date)

        # 8. 收集分红送股数据
        self.collect_dividend()

        # 9. 收集券商一致预期数据
        self.collect_report_rc(start_date=start_date, end_date=end_date)

        print("\n" + "=" * 80)
        print("全量数据收集完成")
        print("=" * 80)

    def incremental_update(self):
        """增量更新数据"""
        print("\n" + "=" * 80)
        print("开始增量更新")
        print("=" * 80)

        # 1. 更新股票基本信息
        self.collect_stock_basic()

        # 2. 增量更新日线数据（前复权）
        self.collect_daily_price(incremental=True)

        # 3. 增量更新每日指标（市值/PE/换手率等，查 daily_basic 自身最新日期）
        self.collect_daily_basic(incremental=True)

        # 4. 增量更新资金流向（查 moneyflow 自身最新日期）
        self.collect_moneyflow(incremental=True)

        # 5. 增量更新指数日线（查 index_daily 自身最新日期，仅 6 个指数）
        self.collect_index_data(incremental=True)

        # 6. 增量更新财务报表（最近 90 天，捕获新季报披露）
        recent_start = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        self.collect_financial_data(start_date=recent_start)

        # 7. 增量更新财务指标（最近 90 天）
        self.collect_fina_indicator(start_date=recent_start)

        # 8. 增量更新券商预测数据（最近3个月）
        self.collect_report_rc(incremental=True)

        print("\n" + "=" * 80)
        print("增量更新完成")
        print("=" * 80)

    def collect_daily_basic_single(self, ts_code: str, start_date: str, end_date: str) -> int:
        """
        收集单只股票的每日指标数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            收集的记录数
        """
        try:
            daily_basic_df = self.api.get_daily_basic(ts_code, start_date, end_date)

            if daily_basic_df.empty:
                return 0

            db = DatabaseManager()
            db.insert_dataframe('daily_basic', daily_basic_df)
            db.close()

            return len(daily_basic_df)

        except Exception as e:
            print(f"收集 {ts_code} 每日指标失败: {e}")
            return 0

    def collect_daily_basic(self, start_date: str = None, end_date: str = None,
                           ts_codes: List[str] = None, incremental: bool = False):
        """
        收集每日指标数据（市值、估值等）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            ts_codes: 股票代码列表
            incremental: 是否增量更新
        """
        print("\n" + "=" * 80)
        print("开始收集每日指标数据（市值、估值等）...")
        print("=" * 80)

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            if incremental:
                latest_date = self.db.get_latest_trade_date(table='daily_basic')
                if latest_date:
                    start_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
                else:
                    start_date = config.start_date
            else:
                start_date = config.start_date

        print(f"日期范围: {start_date} - {end_date}")

        if ts_codes is None:
            stock_df = self.db.get_stock_list()
            if stock_df.empty:
                print("股票列表为空，请先收集股票基本信息")
                return
            ts_codes = stock_df['ts_code'].tolist()

        print(f"待收集股票数: {len(ts_codes)}")

        total_records = 0
        failed_stocks = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_code = {
                executor.submit(self.collect_daily_basic_single, code, start_date, end_date): code
                for code in ts_codes
            }

            with tqdm(total=len(ts_codes), desc="收集每日指标") as pbar:
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        records = future.result()
                        total_records += records
                    except Exception as e:
                        print(f"\n{code} 收集失败: {e}")
                        failed_stocks.append(code)
                    finally:
                        pbar.update(1)

        update_type = 'incremental' if incremental else 'full'
        status = 'success' if not failed_stocks else 'partial'
        error_msg = f"失败股票: {','.join(failed_stocks[:10])}" if failed_stocks else None

        self.db.log_update(
            table_name='daily_basic',
            update_type=update_type,
            start_date=start_date,
            end_date=end_date,
            records_count=total_records,
            status=status,
            error_message=error_msg
        )

        print(f"\n每日指标数据收集完成: 总记录数 {total_records}")
        if failed_stocks:
            print(f"失败股票数: {len(failed_stocks)}")

    def collect_fina_indicator_single(self, ts_code: str, start_date: str, end_date: str) -> int:
        """
        收集单只股票的财务指标数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            收集的记录数
        """
        try:
            fina_df = self.api.get_fina_indicator(ts_code, start_date, end_date)

            if fina_df.empty:
                return 0

            db = DatabaseManager()
            db.insert_dataframe('fina_indicator', fina_df)
            db.close()

            return len(fina_df)

        except Exception as e:
            print(f"收集 {ts_code} 财务指标失败: {e}")
            return 0

    def collect_fina_indicator(self, start_date: str = None, end_date: str = None,
                               ts_codes: List[str] = None):
        """
        收集财务指标数据（ROE、ROA等）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            ts_codes: 股票代码列表
        """
        print("\n" + "=" * 80)
        print("开始收集财务指标数据（ROE、ROA等）...")
        print("=" * 80)

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            start_date = config.start_date

        print(f"日期范围: {start_date} - {end_date}")

        if ts_codes is None:
            stock_df = self.db.get_stock_list()
            if stock_df.empty:
                print("股票列表为空，请先收集股票基本信息")
                return
            ts_codes = stock_df['ts_code'].tolist()

        print(f"待收集股票数: {len(ts_codes)}")

        total_records = 0
        failed_stocks = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_code = {
                executor.submit(self.collect_fina_indicator_single, code, start_date, end_date): code
                for code in ts_codes
            }

            with tqdm(total=len(ts_codes), desc="收集财务指标") as pbar:
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        records = future.result()
                        total_records += records
                    except Exception as e:
                        print(f"\n{code} 收集失败: {e}")
                        failed_stocks.append(code)
                    finally:
                        pbar.update(1)

        status = 'success' if not failed_stocks else 'partial'
        error_msg = f"失败股票: {','.join(failed_stocks[:10])}" if failed_stocks else None

        self.db.log_update(
            table_name='fina_indicator',
            update_type='full',
            start_date=start_date,
            end_date=end_date,
            records_count=total_records,
            status=status,
            error_message=error_msg
        )

        print(f"\n财务指标数据收集完成: 总记录数 {total_records}")
        if failed_stocks:
            print(f"失败股票数: {len(failed_stocks)}")

    def collect_moneyflow_single(self, ts_code: str, start_date: str, end_date: str) -> int:
        """
        收集单只股票的资金流向数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            收集的记录数
        """
        try:
            moneyflow_df = self.api.get_moneyflow(ts_code, start_date, end_date)

            if moneyflow_df.empty:
                return 0

            db = DatabaseManager()
            db.insert_dataframe('moneyflow', moneyflow_df)
            db.close()

            return len(moneyflow_df)

        except Exception as e:
            print(f"收集 {ts_code} 资金流向失败: {e}")
            return 0

    def collect_moneyflow(self, start_date: str = None, end_date: str = None,
                         ts_codes: List[str] = None, incremental: bool = False):
        """
        收集资金流向数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            ts_codes: 股票代码列表
            incremental: 是否增量更新
        """
        print("\n" + "=" * 80)
        print("开始收集资金流向数据...")
        print("=" * 80)

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            if incremental:
                latest_date = self.db.get_latest_trade_date(table='moneyflow')
                if latest_date:
                    start_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
                else:
                    start_date = config.start_date
            else:
                start_date = config.start_date

        print(f"日期范围: {start_date} - {end_date}")

        if ts_codes is None:
            stock_df = self.db.get_stock_list()
            if stock_df.empty:
                print("股票列表为空，请先收集股票基本信息")
                return
            ts_codes = stock_df['ts_code'].tolist()

        print(f"待收集股票数: {len(ts_codes)}")

        total_records = 0
        failed_stocks = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_code = {
                executor.submit(self.collect_moneyflow_single, code, start_date, end_date): code
                for code in ts_codes
            }

            with tqdm(total=len(ts_codes), desc="收集资金流向") as pbar:
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        records = future.result()
                        total_records += records
                    except Exception as e:
                        print(f"\n{code} 收集失败: {e}")
                        failed_stocks.append(code)
                    finally:
                        pbar.update(1)

        update_type = 'incremental' if incremental else 'full'
        status = 'success' if not failed_stocks else 'partial'
        error_msg = f"失败股票: {','.join(failed_stocks[:10])}" if failed_stocks else None

        self.db.log_update(
            table_name='moneyflow',
            update_type=update_type,
            start_date=start_date,
            end_date=end_date,
            records_count=total_records,
            status=status,
            error_message=error_msg
        )

        print(f"\n资金流向数据收集完成: 总记录数 {total_records}")
        if failed_stocks:
            print(f"失败股票数: {len(failed_stocks)}")

    def collect_index_data(self, start_date: str = None, end_date: str = None,
                          incremental: bool = False):
        """
        收集指数数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            incremental: 是否增量更新（从 index_daily 最新日期续采）
        """
        print("\n" + "=" * 80)
        print("开始收集指数数据...")
        print("=" * 80)

        # 1. 收集指数基本信息
        try:
            print("收集指数基本信息...")
            index_basic_df = self.api.get_index_basic(market='')
            if not index_basic_df.empty:
                self.db.insert_dataframe('index_basic', index_basic_df, mode='replace')
                print(f"指数基本信息收集完成: {len(index_basic_df)} 条")
        except Exception as e:
            print(f"收集指数基本信息失败: {e}")
            return

        # 2. 收集主要指数的日线数据
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            if incremental:
                latest_date = self.db.get_latest_trade_date(table='index_daily')
                if latest_date:
                    start_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
                else:
                    start_date = config.start_date
            else:
                start_date = config.start_date

        print(f"日期范围: {start_date} - {end_date}")

        # 主要指数列表
        major_indices = [
            '000001.SH',  # 上证指数
            '000300.SH',  # 沪深300
            '000905.SH',  # 中证500
            '000852.SH',  # 中证1000
            '399001.SZ',  # 深证成指
            '399006.SZ',  # 创业板指
        ]

        print(f"收集 {len(major_indices)} 个主要指数的日线数据...")

        total_records = 0
        for ts_code in tqdm(major_indices, desc="收集指数日线"):
            try:
                index_daily_df = self.api.get_index_daily(ts_code, start_date, end_date)
                if not index_daily_df.empty:
                    self.db.insert_dataframe('index_daily', index_daily_df)
                    total_records += len(index_daily_df)
            except Exception as e:
                print(f"\n收集 {ts_code} 失败: {e}")

        self.db.log_update(
            table_name='index_daily',
            update_type='full',
            start_date=start_date,
            end_date=end_date,
            records_count=total_records,
            status='success'
        )

        print(f"\n指数日线数据收集完成: 总记录数 {total_records}")

    def collect_dividend_single(self, ts_code: str) -> int:
        """
        收集单只股票的分红数据

        Args:
            ts_code: 股票代码

        Returns:
            收集的记录数
        """
        try:
            dividend_df = self.api.get_dividend(ts_code)

            if dividend_df.empty:
                return 0

            db = DatabaseManager()
            db.insert_dataframe('dividend', dividend_df)
            db.close()

            return len(dividend_df)

        except Exception as e:
            print(f"收集 {ts_code} 分红数据失败: {e}")
            return 0

    def collect_dividend(self, ts_codes: List[str] = None):
        """
        收集分红送股数据

        Args:
            ts_codes: 股票代码列表
        """
        print("\n" + "=" * 80)
        print("开始收集分红送股数据...")
        print("=" * 80)

        if ts_codes is None:
            stock_df = self.db.get_stock_list()
            if stock_df.empty:
                print("股票列表为空，请先收集股票基本信息")
                return
            ts_codes = stock_df['ts_code'].tolist()

        print(f"待收集股票数: {len(ts_codes)}")

        total_records = 0
        failed_stocks = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_code = {
                executor.submit(self.collect_dividend_single, code): code
                for code in ts_codes
            }

            with tqdm(total=len(ts_codes), desc="收集分红数据") as pbar:
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        records = future.result()
                        total_records += records
                    except Exception as e:
                        print(f"\n{code} 收集失败: {e}")
                        failed_stocks.append(code)
                    finally:
                        pbar.update(1)

        status = 'success' if not failed_stocks else 'partial'
        error_msg = f"失败股票: {','.join(failed_stocks[:10])}" if failed_stocks else None

        self.db.log_update(
            table_name='dividend',
            update_type='full',
            start_date='',
            end_date='',
            records_count=total_records,
            status=status,
            error_message=error_msg
        )

        print(f"\n分红数据收集完成: 总记录数 {total_records}")
        if failed_stocks:
            print(f"失败股票数: {len(failed_stocks)}")

    def collect_report_rc_single(self, ts_code: str, start_date: str, end_date: str) -> int:
        """
        收集单只股票的券商一致预期数据

        Args:
            ts_code: 股票代码
            start_date: 报告发布开始日期
            end_date: 报告发布结束日期

        Returns:
            收集的记录数
        """
        try:
            report_rc_df = self.api.get_report_rc(ts_code, start_date, end_date)

            if report_rc_df.empty:
                return 0

            db = DatabaseManager()
            db.insert_dataframe('report_rc', report_rc_df)
            db.close()

            return len(report_rc_df)

        except Exception as e:
            print(f"收集 {ts_code} 券商预测失败: {e}")
            return 0

    def collect_report_rc(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                         ts_codes: List[str] = None, incremental: bool = False):
        """
        收集券商一致预期数据（分析师预测净利润）

        Args:
            start_date: 报告发布开始日期
            end_date: 报告发布结束日期
            ts_codes: 股票代码列表
            incremental: 是否增量更新（只获取最近3个月的数据）
        """
        print("\n" + "=" * 80)
        print("开始收集券商一致预期数据（分析师预测净利润）...")
        print("=" * 80)

        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        if start_date is None:
            if incremental:
                # 增量更新：获取最近3个月的数据
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
                print("增量更新模式：获取最近3个月的数据")
            else:
                start_date = config.start_date

        print(f"日期范围: {start_date} - {end_date}")

        if ts_codes is None:
            stock_df = self.db.get_stock_list()
            if stock_df.empty:
                print("股票列表为空，请先收集股票基本信息")
                return
            ts_codes = stock_df['ts_code'].tolist()

        print(f"待收集股票数: {len(ts_codes)}")

        total_records = 0
        failed_stocks = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_code = {
                executor.submit(self.collect_report_rc_single, code, start_date, end_date): code
                for code in ts_codes
            }

            with tqdm(total=len(ts_codes), desc="收集券商预测") as pbar:
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        records = future.result()
                        total_records += records
                    except Exception as e:
                        print(f"\n{code} 收集失败: {e}")
                        failed_stocks.append(code)
                    finally:
                        pbar.update(1)

        status = 'success' if not failed_stocks else 'partial'
        error_msg = f"失败股票: {','.join(failed_stocks[:10])}" if failed_stocks else None

        self.db.log_update(
            table_name='report_rc',
            update_type='incremental' if incremental else 'full',
            start_date=start_date,
            end_date=end_date,
            records_count=total_records,
            status=status,
            error_message=error_msg
        )

        print(f"\n券商预测数据收集完成: 总记录数 {total_records}")
        if failed_stocks:
            print(f"失败股票数: {len(failed_stocks)}")

    def collect_moneyflow_fast(self, start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               resume: bool = True, workers: int = 5):
        """
        并发全量收集资金流向数据（含断点续传）

        适用于首次全量建库或需要重新收集所有股票的场景。
        增量更新请使用 collect_moneyflow(incremental=True)。

        Args:
            start_date: 开始日期 (YYYYMMDD)，默认近10年
            end_date: 结束日期 (YYYYMMDD)，默认今日
            resume: 是否从上次中断处继续（读取断点文件）
            workers: 并发线程数
        """
        import sys
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        if not start_date or not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=10 * 365)).strftime('%Y%m%d')

        print(f"\n开始收集资金流向数据: {start_date} 到 {end_date}")
        print(f"优化模式: 并发处理 ({workers} 线程) + 批量插入 + 断点续传")

        stocks = self.db.get_stock_list()
        total_stocks = len(stocks)
        print(f"共 {total_stocks} 只股票\n")

        progress = _load_moneyflow_progress() if resume else {'completed_stocks': [], 'last_index': 0}
        start_idx = progress['last_index']
        completed_set = set(progress['completed_stocks'])

        if start_idx > 0:
            print(f"从第 {start_idx + 1} 只股票继续收集...")
            print(f"已完成: {len(completed_set)} 只股票\n")

        success_count = len(completed_set)
        error_count = 0
        total_records = 0
        batch_records = []
        lock = threading.Lock()

        stocks_to_process = []
        for idx in range(start_idx, total_stocks):
            row = stocks.iloc[idx]
            ts_code = row['ts_code']
            if ts_code not in completed_set:
                stocks_to_process.append((idx, ts_code, row['name']))

        print(f"待处理股票数: {len(stocks_to_process)}\n")

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                api_instances = [TushareAPI() for _ in range(workers)]
                future_to_stock = {}
                for i, (idx, ts_code, name) in enumerate(stocks_to_process):
                    api = api_instances[i % workers]
                    future = executor.submit(_fetch_stock_moneyflow,
                                            ts_code, name, start_date, end_date, api)
                    future_to_stock[future] = (idx, ts_code, name)

                for future in as_completed(future_to_stock):
                    idx, ts_code, name = future_to_stock[future]
                    _, df, error = future.result()

                    with lock:
                        if error:
                            error_count += 1
                            if error_count <= 10:
                                print(f"✗ [{idx+1}/{total_stocks}] {ts_code} {name} 失败: {error}", flush=True)
                        elif df is not None:
                            batch_records.append(df)
                            total_records += len(df)
                            success_count += 1
                            completed_set.add(ts_code)

                            if success_count % 10 == 0 or success_count < 20:
                                pct = (idx + 1) / total_stocks * 100
                                print(f"[{idx+1}/{total_stocks}] {pct:.1f}% | {ts_code} {name} | 成功: {success_count}", flush=True)

                            if len(batch_records) >= 50:
                                combined_df = pd.concat(batch_records, ignore_index=True)
                                self.db.insert_dataframe('moneyflow', combined_df)
                                batch_records = []
                                _save_moneyflow_progress({
                                    'completed_stocks': list(completed_set),
                                    'last_index': idx + 1
                                })
                                print(f">>> 批次完成 | 成功: {success_count} | 失败: {error_count} | 总记录: {total_records}", flush=True)

            if batch_records:
                combined_df = pd.concat(batch_records, ignore_index=True)
                self.db.insert_dataframe('moneyflow', combined_df)

            print(f"\n{'='*60}")
            print(f"收集完成!")
            print(f"成功: {success_count} 只股票")
            print(f"失败: {error_count} 只股票")
            print(f"总记录数: {total_records}")
            print(f"{'='*60}")

            progress_file = Path('data/moneyflow_progress.json')
            if progress_file.exists():
                progress_file.unlink()

            self.db.log_update(
                table_name='moneyflow',
                update_type='full',
                start_date=start_date,
                end_date=end_date,
                records_count=total_records,
                status='success' if error_count == 0 else 'partial'
            )

        except KeyboardInterrupt:
            print("\n\n用户中断，保存进度...")
            _save_moneyflow_progress({
                'completed_stocks': list(completed_set),
                'last_index': max(
                    (idx for idx, ts_code, _ in stocks_to_process if ts_code in completed_set),
                    default=start_idx
                )
            })
            print("进度已保存")

        except Exception as e:
            print(f"收集失败: {e}")
            import traceback
            traceback.print_exc()
            self.db.log_update(
                table_name='moneyflow',
                update_type='full',
                start_date=start_date or '',
                end_date=end_date or '',
                records_count=0,
                status='failed',
                error_message=str(e)
            )

    def collect_holder_data(self, start_year: int = 2016, end_year: int = None):
        """
        按季度末批量收集股东户数 (stk_holdernumber) 并入库
        约 40 次 API 请求，耗时几分钟
        """
        import duckdb as _duckdb
        from tqdm import tqdm as _tqdm
        import pandas as _pd
        from datetime import datetime as _dt

        if end_year is None:
            end_year = _dt.now().year

        CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS stk_holdernumber (
            ts_code    VARCHAR NOT NULL,
            ann_date   VARCHAR,
            end_date   VARCHAR NOT NULL,
            holder_num BIGINT,
            PRIMARY KEY (ts_code, end_date)
        )"""

        db_path = self.db.db_path

        with _duckdb.connect(db_path) as conn:
            conn.execute(CREATE_SQL)
            existing = set(conn.execute(
                "SELECT DISTINCT end_date FROM stk_holdernumber"
            ).fetchdf()["end_date"].tolist())

        dates = []
        for y in range(start_year, end_year + 1):
            for mmdd in ["0331", "0630", "0930", "1231"]:
                d = f"{y}{mmdd}"
                if d <= _dt.now().strftime("%Y%m%d"):
                    dates.append(d)
        todo = [d for d in dates if d not in existing]
        print(f"待收集季度: {len(todo)} 个（已有: {len(existing)} 个）")
        if not todo:
            print("stk_holdernumber 已是最新，无需收集")
            return

        total_rows = 0
        with _duckdb.connect(db_path) as conn:
            for end_date in _tqdm(todo, desc="收集股东户数（按季度）"):
                try:
                    df = self.api._retry_request(
                        self.api.pro.stk_holdernumber,
                        end_date=end_date,
                        fields="ts_code,ann_date,end_date,holder_num"
                    )
                    if df is None or df.empty:
                        continue
                    df["holder_num"] = _pd.to_numeric(df["holder_num"], errors="coerce")
                    df = df.dropna(subset=["ts_code", "end_date", "holder_num"])
                    df["holder_num"] = df["holder_num"].astype(int)
                    if not df.empty:
                        conn.execute("INSERT OR REPLACE INTO stk_holdernumber SELECT * FROM df")
                        total_rows += len(df)
                except Exception as e:
                    print(f"  [{end_date}] 失败: {e}")

        print(f"完成！共入库 {total_rows:,} 条记录")
        with _duckdb.connect(db_path, read_only=True) as conn:
            r = conn.execute("""
                SELECT COUNT(*) AS rows, COUNT(DISTINCT ts_code) AS stocks,
                       MIN(end_date) AS min_date, MAX(end_date) AS max_date
                FROM stk_holdernumber
            """).fetchone()
        print(f"  stk_holdernumber: {r[0]:,} 行, {r[1]} 只, {r[2]}~{r[3]}")
