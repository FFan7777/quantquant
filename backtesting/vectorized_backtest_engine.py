"""
向量化回测引擎

使用 NumPy 和 Pandas 向量化操作加速回测，避免低效的 Python 循环
主要优化点：
1. 预计算所有交易日的价格矩阵
2. 向量化计算持仓市值和收益
3. 批量处理交易成本
4. 减少数据复制和内存分配
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class VectorizedBacktestEngine:
    """向量化回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,
        transfer_fee_rate: float = 0.00002,
        slippage_rate: float = 0.0001,
        min_commission: float = 5.0,
        min_trade_value: float = 1000.0,   # 低于此金额的交易跳过（消除漂移修正噪声）
    ):
        """
        初始化回测引擎

        参数:
            initial_capital: 初始资金
            commission_rate: 佣金费率（买卖双向，默认万三）
            stamp_tax_rate: 印花税率（仅卖出，默认千一）
            transfer_fee_rate: 过户费率（买卖双向，默认万0.2）
            slippage_rate: 滑点费率（买卖双向）
            min_commission: 最小佣金
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission
        self.min_trade_value = min_trade_value

        # 回测结果
        self.equity_curve = []
        self.trades = []
        self.total_commission = 0.0
        self.total_stamp_tax = 0.0
        self.total_transfer_fee = 0.0
        self.total_slippage = 0.0

        # 缓存数据
        self.price_matrix = None
        self.stock_list = None
        self.date_list = None

    def prepare_data(self, data: pd.DataFrame) -> None:
        """
        预处理数据，构建价格矩阵（向量化优化）

        参数:
            data: 价格数据，需包含 ts_code, trade_date, close 列
        """
        # 透视表：日期 x 股票代码
        self.price_matrix = data.pivot(
            index='trade_date',
            columns='ts_code',
            values='close'
        )

        self.date_list = self.price_matrix.index.tolist()
        self.stock_list = self.price_matrix.columns.tolist()

        print(f"✓ 价格矩阵构建完成: {len(self.date_list)} 天 x {len(self.stock_list)} 只股票")

    def calculate_transaction_cost_vectorized(
        self,
        trade_values: np.ndarray,
        is_sell: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        向量化计算交易成本

        参数:
            trade_values: 交易金额数组
            is_sell: 是否卖出的布尔数组

        返回:
            (总佣金, 总印花税, 总过户费, 总滑点)
        """
        # 佣金（买卖双向）
        commissions = np.maximum(trade_values * self.commission_rate, self.min_commission)
        total_commission = commissions.sum()

        # 印花税（仅卖出）
        stamp_taxes = trade_values * self.stamp_tax_rate * is_sell
        total_stamp_tax = stamp_taxes.sum()

        # 过户费（买卖双向）
        transfer_fees = trade_values * self.transfer_fee_rate
        total_transfer_fee = transfer_fees.sum()

        # 滑点（买卖双向）
        slippages = trade_values * self.slippage_rate
        total_slippage = slippages.sum()

        return total_commission, total_stamp_tax, total_transfer_fee, total_slippage

    def rebalance_portfolio_vectorized(
        self,
        current_positions: np.ndarray,
        target_weights: Dict[str, float],
        current_prices: pd.Series,
        current_capital: float,
        date: str,
        date_idx: int
    ) -> Tuple[np.ndarray, float, List[Dict]]:
        """
        向量化调整投资组合

        参数:
            current_positions: 当前持仓数组（与 stock_list 对应）
            target_weights: 目标持仓权重 {ts_code: weight}
            current_prices: 当前价格 Series
            current_capital: 当前资金
            date: 当前日期
            date_idx: 日期索引

        返回:
            (新持仓数组, 剩余资金, 交易记录)
        """
        trades = []

        # 获取有效价格的股票索引
        valid_prices = current_prices.notna()

        # 计算当前持仓市值（向量化）
        position_values = current_positions * current_prices.fillna(0).values
        total_value = current_capital + position_values.sum()

        # 构建目标持仓数组
        target_position_array = np.zeros(len(self.stock_list))
        for ts_code, weight in target_weights.items():
            if ts_code in self.stock_list:
                idx = self.stock_list.index(ts_code)
                price = current_prices.iloc[idx]
                if valid_prices.iloc[idx] and not pd.isna(price) and price > 0:
                    target_value = total_value * weight
                    target_position_array[idx] = int(target_value / price)

        # 计算持仓变化（向量化）
        position_diff = target_position_array - current_positions

        # 分离买入和卖出
        sell_mask = position_diff < 0
        buy_mask = position_diff > 0

        # 先处理卖出（向量化）
        if sell_mask.any():
            sell_shares = np.abs(position_diff[sell_mask])
            sell_prices = current_prices.values[sell_mask]
            # 过滤 NaN 价格 + 过滤小额漂移修正（< min_trade_value）
            valid_sell = ~np.isnan(sell_prices) & (sell_shares * np.where(np.isnan(sell_prices), 0, sell_prices) >= self.min_trade_value)
            if valid_sell.any():
                sell_shares = sell_shares[valid_sell]
                sell_prices = sell_prices[valid_sell]
            else:
                sell_shares = np.array([])
                sell_prices = np.array([])
            sell_values = sell_shares * sell_prices

            if len(sell_values) > 0:
                # 计算卖出成本
                commission, stamp_tax, transfer_fee, slippage = self.calculate_transaction_cost_vectorized(
                    sell_values,
                    np.ones(len(sell_values), dtype=bool)
                )

                # 更新资金
                current_capital += sell_values.sum() - (commission + stamp_tax + transfer_fee + slippage)

                # 累计成本
                self.total_commission += commission
                self.total_stamp_tax += stamp_tax
                self.total_transfer_fee += transfer_fee
                self.total_slippage += slippage

                # 记录交易
                sell_indices = np.where(sell_mask)[0]
                if valid_sell.any():
                    sell_indices = sell_indices[valid_sell]
                for i, idx in enumerate(sell_indices):
                    trades.append({
                        'date': date,
                        'ts_code': self.stock_list[idx],
                        'action': 'sell',
                        'shares': sell_shares[i],
                        'price': sell_prices[i],
                        'value': sell_values[i]
                    })

        # 再处理买入（向量化）
        if buy_mask.any():
            buy_shares = position_diff[buy_mask]
            buy_prices = current_prices.values[buy_mask]
            # 过滤 NaN 价格 + 过滤小额漂移修正（< min_trade_value）
            valid_buy = ~np.isnan(buy_prices) & (buy_shares * np.where(np.isnan(buy_prices), 0, buy_prices) >= self.min_trade_value)
            if valid_buy.any():
                buy_shares = buy_shares[valid_buy]
                buy_prices = buy_prices[valid_buy]
            else:
                buy_shares = np.array([])
                buy_prices = np.array([])
            buy_values = buy_shares * buy_prices

            if len(buy_values) > 0:
                # 计算买入成本
                commission, stamp_tax, transfer_fee, slippage = self.calculate_transaction_cost_vectorized(
                    buy_values,
                    np.zeros(len(buy_values), dtype=bool)
                )

                total_buy_cost = buy_values.sum() + commission + stamp_tax + transfer_fee + slippage

                # 检查资金是否充足
                if current_capital >= total_buy_cost:
                    current_capital -= total_buy_cost

                    # 累计成本
                    self.total_commission += commission
                    self.total_stamp_tax += stamp_tax
                    self.total_transfer_fee += transfer_fee
                    self.total_slippage += slippage

                    # 记录交易
                    buy_indices = np.where(buy_mask)[0]
                    if valid_buy.any():
                        buy_indices = buy_indices[valid_buy]
                    for i, idx in enumerate(buy_indices):
                        trades.append({
                            'date': date,
                            'ts_code': self.stock_list[idx],
                            'action': 'buy',
                            'shares': buy_shares[i],
                            'price': buy_prices[i],
                            'value': buy_values[i]
                        })
                else:
                    # 资金不足，按比例缩减买入
                    if total_buy_cost > 0:
                        scale = current_capital / total_buy_cost
                        buy_shares = (buy_shares * scale).astype(int)
                        buy_values = buy_shares * buy_prices

                        commission, stamp_tax, transfer_fee, slippage = self.calculate_transaction_cost_vectorized(
                            buy_values,
                            np.zeros(len(buy_values), dtype=bool)
                        )

                        current_capital -= buy_values.sum() + commission + stamp_tax + transfer_fee + slippage

                        self.total_commission += commission
                        self.total_stamp_tax += stamp_tax
                        self.total_transfer_fee += transfer_fee
                        self.total_slippage += slippage

                        # 更新目标持仓
                        buy_indices = np.where(buy_mask)[0]
                        if valid_buy.any():
                            buy_indices = buy_indices[valid_buy]
                        for i, idx in enumerate(buy_indices):
                            target_position_array[idx] = current_positions[idx] + buy_shares[i]
                            if buy_shares[i] > 0:
                                trades.append({
                                    'date': date,
                                    'ts_code': self.stock_list[idx],
                                    'action': 'buy',
                                    'shares': buy_shares[i],
                                    'price': buy_prices[i],
                                    'value': buy_values[i]
                                })

        return target_position_array, current_capital, trades

    def run(
        self,
        strategy,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        运行回测（向量化优化版本）

        参数:
            strategy: 策略对象
            data: 价格数据
            start_date: 开始日期
            end_date: 结束日期

        返回:
            回测结果 DataFrame
        """
        print(f"\n{'='*60}")
        print(f"开始回测: {strategy.name} (向量化版本)")
        print(f"{'='*60}")

        # 数据预处理
        data = data.copy()
        data['trade_date'] = data['trade_date'].astype(str)
        data = data.sort_values(['trade_date', 'ts_code'])

        if start_date:
            data = data[data['trade_date'] >= start_date]
        if end_date:
            data = data[data['trade_date'] <= end_date]

        # 构建价格矩阵
        self.prepare_data(data)

        print(f"回测期间: {self.date_list[0]} - {self.date_list[-1]}")
        print(f"交易日数: {len(self.date_list)}")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print()

        # 初始化
        current_capital = self.initial_capital
        current_positions = np.zeros(len(self.stock_list))
        self.equity_curve = []
        self.trades = []

        prev_total_value = self.initial_capital

        # 逐日回测
        for i, date in enumerate(self.date_list):
            # 获取当日价格（向量化）
            current_prices = self.price_matrix.loc[date]

            # 生成交易信号
            target_weights = strategy.on_bar(data, date)

            # 调整持仓
            if target_weights:
                current_positions, current_capital, trades = self.rebalance_portfolio_vectorized(
                    current_positions,
                    target_weights,
                    current_prices,
                    current_capital,
                    date,
                    i
                )
                self.trades.extend(trades)

            # 计算当前总资产（向量化）
            position_value = (current_positions * current_prices.fillna(0).values).sum()
            total_value = current_capital + position_value

            # 计算每日收益率
            daily_return = (total_value - prev_total_value) / prev_total_value if prev_total_value > 0 else 0
            prev_total_value = total_value

            # 记录权益曲线
            self.equity_curve.append({
                'date': date,
                'capital': current_capital,
                'position_value': position_value,
                'total_value': total_value,
                'daily_return': daily_return
            })

            # 显示进度
            if (i + 1) % 100 == 0 or i == len(self.date_list) - 1:
                progress = (i + 1) / len(self.date_list) * 100
                print(f"\r进度: {progress:.1f}% ({i+1}/{len(self.date_list)}) "
                      f"总资产: {total_value:,.2f} "
                      f"收益率: {(total_value/self.initial_capital - 1)*100:.2f}%", end='')

        print("\n")

        # 打印交易成本统计
        print(f"交易成本统计:")
        print(f"  总佣金: {self.total_commission:,.2f}")
        print(f"  总印花税: {self.total_stamp_tax:,.2f}")
        print(f"  总过户费: {self.total_transfer_fee:,.2f}")
        print(f"  总滑点: {self.total_slippage:,.2f}")
        total_cost = self.total_commission + self.total_stamp_tax + self.total_transfer_fee + self.total_slippage
        print(f"  总成本: {total_cost:,.2f}")
        print(f"  成本占初始资金比例: {total_cost/self.initial_capital*100:.2f}%")
        print()

        return pd.DataFrame(self.equity_curve)

    def get_trades(self) -> pd.DataFrame:
        """获取交易记录"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
