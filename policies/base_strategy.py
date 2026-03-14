"""
策略基类
"""

from typing import Dict
import pandas as pd


class BaseStrategy:
    """
    策略基类。

    简单策略重写 generate_signals；
    复杂策略（如 EnhancedFundamentalStrategy）可直接重写 on_bar 以获得完全控制权。
    """

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.positions: Dict[str, float] = {}  # 当前持仓 {ts_code: weight}

    def generate_signals(self, data: pd.DataFrame, date: str) -> Dict[str, float]:
        """
        生成交易信号。返回 {ts_code: weight} 目标持仓权重。
        默认返回空仓，子类应重写此方法。
        """
        return {}

    def on_bar(self, data: pd.DataFrame, date: str) -> Dict[str, float]:
        """每个交易日由回测引擎调用，默认委托给 generate_signals。"""
        return self.generate_signals(data, date)

    def update_positions(self, positions: Dict[str, float]) -> None:
        """更新持仓记录"""
        self.positions = positions.copy()
