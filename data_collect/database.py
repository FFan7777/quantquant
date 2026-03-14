"""
数据库管理模块
"""
import duckdb
from typing import Optional, List, Dict, Any
import pandas as pd
from .config import config
from .schema import ALL_SCHEMAS


class DatabaseManager:
    """DuckDB 数据库管理类"""

    def __init__(self, db_path: str = None):
        """
        初始化数据库连接

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path or config.db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self):
        """建立数据库连接"""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            print(f"已连接到数据库: {self.db_path}")

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("数据库连接已关闭")

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()

    def init_tables(self):
        """初始化所有表结构"""
        self.connect()
        for schema in ALL_SCHEMAS:
            self.conn.execute(schema)
        print("数据库表结构初始化完成")

    def insert_dataframe(self, table_name: str, df: pd.DataFrame, mode: str = 'append'):
        """
        将 DataFrame 插入数据库

        Args:
            table_name: 表名
            df: 数据框
            mode: 插入模式，'append' 或 'replace'
        """
        if df.empty:
            print(f"警告: DataFrame 为空，跳过插入到 {table_name}")
            return

        self.connect()

        if mode == 'replace':
            # 先删除表中的数据
            self.conn.execute(f"DELETE FROM {table_name}")

        # 使用 DuckDB 的高效插入
        self.conn.register('temp_df', df)
        columns = ', '.join(df.columns)
        placeholders = ', '.join(['?' for _ in df.columns])

        try:
            # 获取 DataFrame 的列名
            df_columns = ', '.join(df.columns)

            # DuckDB 使用 ON CONFLICT 语法
            if table_name == 'stock_basic':
                # stock_basic 有主键
                if mode == 'replace':
                    # 已经在前面删除了，直接插入
                    self.conn.execute(f"INSERT INTO {table_name} ({df_columns}) SELECT {df_columns} FROM temp_df")
                else:
                    # append 模式，使用 ON CONFLICT
                    self.conn.execute(f"""
                        INSERT INTO {table_name} ({df_columns}) SELECT {df_columns} FROM temp_df
                        ON CONFLICT (ts_code) DO UPDATE SET
                            symbol = EXCLUDED.symbol,
                            name = EXCLUDED.name,
                            area = EXCLUDED.area,
                            industry = EXCLUDED.industry,
                            list_date = EXCLUDED.list_date
                    """)
            elif table_name in ['daily_price', 'daily_basic', 'balance_sheet', 'income_statement', 'cash_flow',
                               'fina_indicator', 'moneyflow', 'index_daily', 'dividend', 'report_rc']:
                # 这些表有复合主键，使用 ON CONFLICT DO NOTHING 避免重复
                # 先获取插入前的记录数
                count_before = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

                self.conn.execute(f"INSERT INTO {table_name} ({df_columns}) SELECT {df_columns} FROM temp_df ON CONFLICT DO NOTHING")

                # 获取插入后的记录数
                count_after = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                inserted = count_after - count_before
                skipped = len(df) - inserted

                if inserted > 0:
                    print(f"成功插入 {inserted} 条记录到 {table_name}" + (f" (跳过 {skipped} 条重复记录)" if skipped > 0 else ""))
                else:
                    print(f"跳过 {skipped} 条重复记录到 {table_name}")
            else:
                # 其他表直接插入
                self.conn.execute(f"INSERT INTO {table_name} ({df_columns}) SELECT {df_columns} FROM temp_df")
                print(f"成功插入 {len(df)} 条记录到 {table_name}")

        except Exception as e:
            print(f"插入数据到 {table_name} 失败: {e}")
            # 不再抛出异常，而是继续执行
            # raise
        finally:
            self.conn.unregister('temp_df')

    def query(self, sql: str) -> pd.DataFrame:
        """
        执行查询并返回 DataFrame

        Args:
            sql: SQL 查询语句

        Returns:
            查询结果
        """
        self.connect()
        return self.conn.execute(sql).fetchdf()

    def execute(self, sql: str):
        """
        执行 SQL 语句

        Args:
            sql: SQL 语句
        """
        self.connect()
        self.conn.execute(sql)

    def get_latest_trade_date(self, ts_code: str = None,
                              table: str = 'daily_price') -> Optional[str]:
        """
        获取指定表的最新交易日期

        Args:
            ts_code: 股票代码，如果为 None 则返回所有股票的最新日期
            table: 要查询的表名，默认 daily_price

        Returns:
            最新交易日期
        """
        self.connect()

        if ts_code:
            sql = f"SELECT MAX(trade_date) as max_date FROM {table} WHERE ts_code = '{ts_code}'"
        else:
            sql = f"SELECT MAX(trade_date) as max_date FROM {table}"

        result = self.conn.execute(sql).fetchone()
        return result[0] if result and result[0] else None

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表

        Returns:
            股票列表 DataFrame
        """
        return self.query("SELECT * FROM stock_basic ORDER BY ts_code")

    def log_update(self, table_name: str, update_type: str, start_date: str,
                   end_date: str, records_count: int, status: str,
                   error_message: str = None):
        """
        记录更新日志

        Args:
            table_name: 表名
            update_type: 更新类型
            start_date: 起始日期
            end_date: 结束日期
            records_count: 记录数
            status: 状态
            error_message: 错误信息
        """
        self.connect()

        log_data = pd.DataFrame([{
            'table_name': table_name,
            'update_type': update_type,
            'start_date': start_date,
            'end_date': end_date,
            'records_count': records_count,
            'status': status,
            'error_message': error_message
        }])

        self.conn.register('log_df', log_data)
        self.conn.execute("""
            INSERT INTO update_log (table_name, update_type, start_date, end_date,
                                   records_count, status, error_message)
            SELECT table_name, update_type, start_date, end_date,
                   records_count, status, error_message
            FROM log_df
        """)
        self.conn.unregister('log_df')

    def get_update_logs(self, table_name: str = None, limit: int = 100) -> pd.DataFrame:
        """
        获取更新日志

        Args:
            table_name: 表名，如果为 None 则返回所有表的日志
            limit: 返回记录数限制

        Returns:
            更新日志 DataFrame
        """
        if table_name:
            sql = f"""
                SELECT * FROM update_log
                WHERE table_name = '{table_name}'
                ORDER BY created_at DESC
                LIMIT {limit}
            """
        else:
            sql = f"SELECT * FROM update_log ORDER BY created_at DESC LIMIT {limit}"

        return self.query(sql)
