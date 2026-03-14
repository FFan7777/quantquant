"""
配置文件管理模块
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径，默认为 data_collect/config.yaml
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                'config.yaml'
            )

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的多级键，如 'tushare.token'
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def tushare_token(self) -> str:
        """获取 Tushare Token"""
        return self.get('tushare.token')

    @property
    def use_custom_api(self) -> bool:
        """是否使用自定义 API 地址"""
        return self.get('tushare.use_custom_api', False)

    @property
    def request_interval(self) -> float:
        """获取请求间隔"""
        return self.get('tushare.request_interval', 0.2)

    @property
    def batch_size(self) -> int:
        """获取批量请求大小"""
        return self.get('tushare.batch_size', 100)

    @property
    def max_retries(self) -> int:
        """获取最大重试次数"""
        return self.get('tushare.max_retries', 3)

    @property
    def retry_delay(self) -> int:
        """获取重试延迟"""
        return self.get('tushare.retry_delay', 5)

    @property
    def db_path(self) -> str:
        """获取数据库路径"""
        db_path = self.get('database.db_path', 'data/quant.duckdb')
        # 确保数据目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        return db_path

    @property
    def start_date(self) -> str:
        """获取历史数据起始日期"""
        return self.get('data_collection.start_date', '20140101')

    @property
    def include_delisted(self) -> bool:
        """是否包含退市股票"""
        return self.get('data_collection.include_delisted', False)

    @property
    def max_workers(self) -> int:
        """获取最大并发数"""
        return self.get('data_collection.max_workers', 5)

    @property
    def cagr_method(self) -> str:
        """获取 CAGR 计算方式"""
        return self.get('stock_selection.cagr_method', 'historical')


# 全局配置实例
config = Config()
