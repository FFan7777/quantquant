"""
性能指标计算模块

计算回测的各项性能指标：
- 总收益率
- 年化收益率
- 最大回撤
- 夏普比率
- 索提诺比率
- 卡玛比率
- 胜率
- 盈亏比
等
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class PerformanceMetrics:
    """性能指标计算器"""

    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化

        参数:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate

    def calculate_total_return(self, equity_curve: pd.DataFrame) -> float:
        """
        计算总收益率

        参数:
            equity_curve: 权益曲线

        返回:
            总收益率
        """
        if len(equity_curve) == 0:
            return 0.0

        initial_value = equity_curve.iloc[0]['total_value']
        final_value = equity_curve.iloc[-1]['total_value']

        return (final_value - initial_value) / initial_value

    def calculate_annualized_return(
        self,
        equity_curve: pd.DataFrame,
        trading_days_per_year: int = 252
    ) -> float:
        """
        计算年化收益率

        参数:
            equity_curve: 权益曲线
            trading_days_per_year: 每年交易日数

        返回:
            年化收益率
        """
        if len(equity_curve) == 0:
            return 0.0

        total_return = self.calculate_total_return(equity_curve)
        n_days = len(equity_curve)

        if n_days == 0:
            return 0.0

        years = n_days / trading_days_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1

        return annualized_return

    def calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """
        计算最大回撤

        参数:
            equity_curve: 权益曲线

        返回:
            {
                'max_drawdown': 最大回撤,
                'max_drawdown_duration': 最大回撤持续天数,
                'start_date': 回撤开始日期,
                'end_date': 回撤结束日期
            }
        """
        if len(equity_curve) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'start_date': None,
                'end_date': None
            }

        values = equity_curve['total_value'].values
        dates = equity_curve['date'].values

        # 计算累计最高值
        cummax = np.maximum.accumulate(values)

        # 计算回撤
        drawdowns = (values - cummax) / cummax

        # 找到最大回撤
        max_dd_idx = np.argmin(drawdowns)
        max_drawdown = drawdowns[max_dd_idx]

        # 找到最大回撤的起始点
        start_idx = np.argmax(cummax[:max_dd_idx + 1])

        # 计算回撤持续时间
        duration = max_dd_idx - start_idx

        return {
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_duration': duration,
            'start_date': dates[start_idx],
            'end_date': dates[max_dd_idx]
        }

    def calculate_sharpe_ratio(
        self,
        daily_returns: pd.Series,
        trading_days_per_year: int = 252
    ) -> float:
        """
        计算夏普比率

        参数:
            daily_returns: 每日收益率
            trading_days_per_year: 每年交易日数

        返回:
            夏普比率
        """
        if len(daily_returns) == 0:
            return 0.0

        # 计算日收益率的均值和标准差
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()

        if std_return == 0:
            return 0.0

        # 日无风险利率
        daily_rf = self.risk_free_rate / trading_days_per_year

        # 夏普比率
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(trading_days_per_year)

        return sharpe

    def calculate_sortino_ratio(
        self,
        daily_returns: pd.Series,
        trading_days_per_year: int = 252
    ) -> float:
        """
        计算索提诺比率（只考虑下行波动）

        参数:
            daily_returns: 每日收益率
            trading_days_per_year: 每年交易日数

        返回:
            索提诺比率
        """
        if len(daily_returns) == 0:
            return 0.0

        mean_return = daily_returns.mean()

        # 只计算负收益的标准差
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) == 0:
            return 0.0

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0.0

        daily_rf = self.risk_free_rate / trading_days_per_year
        sortino = (mean_return - daily_rf) / downside_std * np.sqrt(trading_days_per_year)

        return sortino

    def calculate_calmar_ratio(
        self,
        equity_curve: pd.DataFrame,
        trading_days_per_year: int = 252
    ) -> float:
        """
        计算卡玛比率（年化收益率 / 最大回撤）

        参数:
            equity_curve: 权益曲线
            trading_days_per_year: 每年交易日数

        返回:
            卡玛比率
        """
        annualized_return = self.calculate_annualized_return(equity_curve, trading_days_per_year)
        max_dd = self.calculate_max_drawdown(equity_curve)['max_drawdown']

        if max_dd == 0:
            return 0.0

        return annualized_return / max_dd

    def calculate_volatility(
        self,
        daily_returns: pd.Series,
        trading_days_per_year: int = 252
    ) -> float:
        """
        计算年化波动率

        参数:
            daily_returns: 每日收益率
            trading_days_per_year: 每年交易日数

        返回:
            年化波动率
        """
        if len(daily_returns) == 0:
            return 0.0

        return daily_returns.std() * np.sqrt(trading_days_per_year)

    def calculate_all_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        trading_days_per_year: int = 252
    ) -> Dict[str, float]:
        """
        计算所有性能指标

        参数:
            equity_curve: 权益曲线
            trades: 交易记录
            trading_days_per_year: 每年交易日数

        返回:
            所有指标的字典
        """
        if len(equity_curve) == 0:
            return {}

        daily_returns = pd.Series(equity_curve['daily_return'].values)

        # 基本指标
        total_return = self.calculate_total_return(equity_curve)
        annualized_return = self.calculate_annualized_return(equity_curve, trading_days_per_year)
        max_dd_info = self.calculate_max_drawdown(equity_curve)

        # 风险调整指标
        sharpe = self.calculate_sharpe_ratio(daily_returns, trading_days_per_year)
        sortino = self.calculate_sortino_ratio(daily_returns, trading_days_per_year)
        calmar = self.calculate_calmar_ratio(equity_curve, trading_days_per_year)

        # 波动率
        volatility = self.calculate_volatility(daily_returns, trading_days_per_year)

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_dd_info['max_drawdown'],
            'max_drawdown_duration': max_dd_info['max_drawdown_duration'],
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'volatility': volatility,
            'initial_value': equity_curve.iloc[0]['total_value'],
            'final_value': equity_curve.iloc[-1]['total_value'],
            'trading_days': len(equity_curve)
        }

        return metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """
        打印性能指标

        参数:
            metrics: 指标字典
        """
        print(f"\n{'='*60}")
        print("回测性能指标")
        print(f"{'='*60}")

        print(f"\n基本指标:")
        print(f"  初始资金: {metrics['initial_value']:,.2f}")
        print(f"  最终资金: {metrics['final_value']:,.2f}")
        print(f"  总收益率: {metrics['total_return']*100:.2f}%")
        print(f"  年化收益率: {metrics['annualized_return']*100:.2f}%")
        print(f"  交易日数: {metrics['trading_days']}")

        print(f"\n风险指标:")
        print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"  最大回撤持续: {metrics['max_drawdown_duration']} 天")
        print(f"  年化波动率: {metrics['volatility']*100:.2f}%")

        print(f"\n风险调整收益:")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  索提诺比率: {metrics['sortino_ratio']:.3f}")
        print(f"  卡玛比率: {metrics['calmar_ratio']:.3f}")

        print(f"\n{'='*60}")
