"""
Tushare API 封装模块
"""
import tushare as ts
import pandas as pd
import time
from typing import Optional, List
from datetime import datetime, timedelta
from .config import config


class TushareAPI:
    """Tushare API 封装类，处理限流和重试"""

    def __init__(self):
        """初始化 Tushare API"""
        ts.set_token(config.tushare_token)
        self.pro = ts.pro_api()

        # 设置自定义 API 地址（用于访问券商一致预期数据）
        if config.use_custom_api:
            self.pro._DataApi__token = config.tushare_token
            self.pro._DataApi__http_url = 'https://tushare.data.godscode.com.cn'

        self.last_request_time = 0
        self.request_interval = config.request_interval

    def _rate_limit(self):
        """限流控制"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.request_interval:
            sleep_time = self.request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _retry_request(self, func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        带重试的请求，支持限流自动重试

        Args:
            func: 请求函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            请求结果
        """
        max_rate_limit_retries = 10  # 限流错误最多重试10次
        rate_limit_retry_count = 0

        for attempt in range(config.max_retries):
            try:
                self._rate_limit()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = str(e)

                # 检查是否是限流错误
                if '每分钟最多访问' in error_msg or 'rate limit' in error_msg.lower():
                    rate_limit_retry_count += 1
                    if rate_limit_retry_count <= max_rate_limit_retries:
                        # 限流错误，等待更长时间后重试
                        wait_time = 60  # 等待60秒
                        print(f"遇到限流错误 (第 {rate_limit_retry_count} 次)，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        # 不计入正常重试次数，继续尝试
                        continue
                    else:
                        print(f"限流错误重试次数已达上限 ({max_rate_limit_retries} 次)")
                        raise

                # 其他错误，使用正常重试逻辑
                print(f"请求失败 (尝试 {attempt + 1}/{config.max_retries}): {e}")
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    print(f"请求最终失败: {e}")
                    raise

        return None

    def get_stock_basic(self, list_status: str = 'L') -> pd.DataFrame:
        """
        获取股票列表

        Args:
            list_status: 上市状态，L=上市 D=退市 P=暂停上市

        Returns:
            股票列表 DataFrame
        """
        return self._retry_request(
            self.pro.stock_basic,
            exchange='',
            list_status=list_status,
            fields='ts_code,symbol,name,area,industry,list_date'
        )

    def get_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            日线数据 DataFrame
        """
        return self._retry_request(
            self.pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

    def get_adj_factor(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取复权因子

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            复权因子 DataFrame
        """
        return self._retry_request(
            self.pro.adj_factor,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

    def get_daily_with_adj(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取带前复权的日线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            前复权日线数据 DataFrame
        """
        # 获取日线数据
        daily_df = self.get_daily(ts_code, start_date, end_date)
        if daily_df is None or daily_df.empty:
            return pd.DataFrame()

        # 获取复权因子
        adj_df = self.get_adj_factor(ts_code, start_date, end_date)
        if adj_df is None or adj_df.empty:
            print(f"警告: {ts_code} 没有复权因子数据，使用原始价格")
            daily_df['adj_factor'] = 1.0
            return daily_df

        # 合并数据
        merged_df = pd.merge(daily_df, adj_df, on=['ts_code', 'trade_date'], how='left')

        # 填充缺失的复权因子（使用前向填充）
        merged_df['adj_factor'] = merged_df['adj_factor'].ffill()
        merged_df['adj_factor'] = merged_df['adj_factor'].fillna(1.0)

        # 计算前复权价格（原始价格 * 复权因子）
        price_columns = ['open', 'high', 'low', 'close', 'pre_close']
        for col in price_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col] * merged_df['adj_factor']

        return merged_df

    def get_balancesheet(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取资产负债表

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            资产负债表 DataFrame
        """
        # 选择关键字段
        fields = [
            'ts_code', 'end_date', 'ann_date', 'f_ann_date', 'report_type', 'comp_type',
            'total_assets', 'total_liab', 'total_hldr_eqy_inc_min_int',
            'total_cur_assets', 'total_nca', 'total_cur_liab', 'total_ncl',
            'money_cap', 'accounts_receiv', 'inventories', 'fix_assets',
            'accounts_pay', 'st_borr', 'lt_borr'
        ]

        return self._retry_request(
            self.pro.balancesheet,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=','.join(fields)
        )

    def get_income(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取利润表

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            利润表 DataFrame
        """
        fields = [
            'ts_code', 'end_date', 'ann_date', 'f_ann_date', 'report_type', 'comp_type',
            'basic_eps', 'diluted_eps', 'total_revenue', 'revenue',
            'operate_profit', 'total_profit', 'n_income', 'n_income_attr_p',
            'oper_cost', 'sell_exp', 'admin_exp', 'fin_exp', 'rd_exp',
            'ebit', 'ebitda'
        ]

        return self._retry_request(
            self.pro.income,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=','.join(fields)
        )

    def get_cashflow(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取现金流量表

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            现金流量表 DataFrame
        """
        fields = [
            'ts_code', 'end_date', 'ann_date', 'f_ann_date', 'report_type', 'comp_type',
            'n_cashflow_act', 'n_cashflow_inv_act', 'n_cash_flows_fnc_act',
            'c_cash_equ_end_period', 'c_cash_equ_beg_period',
            'n_incr_cash_cash_equ', 'free_cashflow'
        ]

        return self._retry_request(
            self.pro.cashflow,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=','.join(fields)
        )

    def get_trade_cal(self, start_date: str, end_date: str, exchange: str = 'SSE') -> pd.DataFrame:
        """
        获取交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所代码

        Returns:
            交易日历 DataFrame
        """
        return self._retry_request(
            self.pro.trade_cal,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date
        )

    def get_daily_basic(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取每日指标（市值、估值等）

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            每日指标 DataFrame
        """
        fields = [
            'ts_code', 'trade_date', 'close',
            'turnover_rate', 'turnover_rate_f', 'volume_ratio',
            'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
            'dv_ratio', 'dv_ttm',
            'total_share', 'float_share', 'free_share',
            'total_mv', 'circ_mv'
        ]

        return self._retry_request(
            self.pro.daily_basic,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=','.join(fields)
        )

    def get_fina_indicator(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取财务指标数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            财务指标 DataFrame
        """
        fields = [
            'ts_code', 'ann_date', 'end_date',
            # 盈利能力
            'roe', 'roe_waa', 'roe_dt', 'roa', 'roic',
            'gross_margin', 'netprofit_margin', 'grossprofit_margin',
            # 营运能力
            'assets_turn', 'ar_turn', 'ca_turn', 'fa_turn', 'turn_days',
            # 偿债能力
            'current_ratio', 'quick_ratio', 'cash_ratio',
            'debt_to_assets', 'debt_to_eqt', 'eqt_to_debt',
            # 成长能力
            'basic_eps_yoy', 'netprofit_yoy', 'op_yoy',
            'roe_yoy', 'tr_yoy', 'or_yoy',
            # 每股指标
            'eps', 'dt_eps', 'bps', 'ocfps', 'cfps',
            # 其他
            'ebit', 'ebitda', 'fcff', 'fcfe', 'profit_dedt'
        ]

        return self._retry_request(
            self.pro.fina_indicator,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=','.join(fields)
        )

    def get_moneyflow(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取资金流向数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            资金流向 DataFrame
        """
        fields = [
            'ts_code', 'trade_date',
            'buy_sm_amount', 'sell_sm_amount',
            'buy_md_amount', 'sell_md_amount',
            'buy_lg_amount', 'sell_lg_amount',
            'buy_elg_amount', 'sell_elg_amount',
            'net_mf_amount'
        ]

        return self._retry_request(
            self.pro.moneyflow,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=','.join(fields)
        )

    # ── 批量按日期接口（增量更新专用，1次API调用取全市场）──────────────────────

    def get_daily_by_date(self, trade_date: str) -> pd.DataFrame:
        """按日期批量获取全市场日线数据（返回原始价格，需手动复权）"""
        return self._retry_request(self.pro.daily, trade_date=trade_date)

    def get_adj_factor_by_date(self, trade_date: str) -> pd.DataFrame:
        """按日期批量获取全市场复权因子"""
        return self._retry_request(self.pro.adj_factor, trade_date=trade_date)

    def get_daily_basic_by_date(self, trade_date: str) -> pd.DataFrame:
        """按日期批量获取全市场每日指标"""
        fields = [
            'ts_code', 'trade_date', 'close',
            'turnover_rate', 'turnover_rate_f', 'volume_ratio',
            'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
            'dv_ratio', 'dv_ttm',
            'total_share', 'float_share', 'free_share',
            'total_mv', 'circ_mv'
        ]
        return self._retry_request(
            self.pro.daily_basic, trade_date=trade_date, fields=','.join(fields)
        )

    def get_moneyflow_by_date(self, trade_date: str) -> pd.DataFrame:
        """按日期批量获取全市场资金流向"""
        fields = [
            'ts_code', 'trade_date',
            'buy_sm_amount', 'sell_sm_amount',
            'buy_md_amount', 'sell_md_amount',
            'buy_lg_amount', 'sell_lg_amount',
            'buy_elg_amount', 'sell_elg_amount',
            'net_mf_amount'
        ]
        return self._retry_request(
            self.pro.moneyflow, trade_date=trade_date, fields=','.join(fields)
        )

    def get_index_basic(self, market: str = '') -> pd.DataFrame:
        """
        获取指数基本信息

        Args:
            market: 市场代码（SSE上交所 SZSE深交所 MSCI MSCI指数 CSI中证指数等）

        Returns:
            指数基本信息 DataFrame
        """
        return self._retry_request(
            self.pro.index_basic,
            market=market
        )

    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数日线数据

        Args:
            ts_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            指数日线 DataFrame
        """
        return self._retry_request(
            self.pro.index_daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

    def get_dividend(self, ts_code: str) -> pd.DataFrame:
        """
        获取分红送股数据

        Args:
            ts_code: 股票代码

        Returns:
            分红送股 DataFrame
        """
        fields = [
            'ts_code', 'end_date', 'ann_date', 'div_proc',
            'stk_div', 'stk_bo_rate', 'stk_co_rate',
            'cash_div', 'cash_div_tax',
            'record_date', 'ex_date', 'pay_date'
        ]

        return self._retry_request(
            self.pro.dividend,
            ts_code=ts_code,
            fields=','.join(fields)
        )

    def get_report_rc(self, ts_code: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取券商一致预期数据（分析师预测净利润）

        Args:
            ts_code: 股票代码
            start_date: 报告发布开始日期（YYYYMMDD）
            end_date: 报告发布结束日期（YYYYMMDD）

        Returns:
            券商预测数据 DataFrame，包含字段：
            - ts_code: 股票代码
            - report_date: 报告发布日期
            - quarter: 预测季度（如 2026Q4）
            - np: 预测净利润（万元）
        """
        # 使用 report_rc 接口获取券商一致预期
        # 只获取关键字段
        fields = [
            'ts_code', 'report_date', 'quarter', 'np'
        ]

        df = self._retry_request(
            self.pro.report_rc,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=','.join(fields)
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # 过滤掉空值
        df = df[df['np'].notna()].copy()

        # 转换净利润单位：万元
        df['np'] = df['np'] / 10000.0

        return df
