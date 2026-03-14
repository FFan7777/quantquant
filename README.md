# A股量化交易数据系统

基于 DuckDB 和 Tushare 的量化交易数据收集与回测系统，当前主策略为增强型基本面策略 v3。

## 核心特性

- **完整数据库** — 6 张核心表 + 7 张辅助表，2.9 GB，覆盖 2014–2026
- **无未来信息** — 使用 `f_ann_date`（第一披露日）作为财务数据可知时点
- **每日动态调仓** — 持仓/出仓条件分离，无固定调仓周期
- **三档仓位控制** — 依上证指数相对 MA20/MA60 自动调节仓位上限
- **向量化回测引擎** — NumPy/Pandas 矩阵化计算，全量 A 股 5 年回测 < 5 秒

## 数据库概览

| 指标 | 值 |
|------|----|
| 数据库大小 | 2.9 GB |
| 总记录数 | 24,000,000+ |
| 股票数量 | 5,484 只 |
| 日期范围 | 2014-01-02 至今 |

## 快速开始

```bash
# 进入项目目录并激活虚拟环境
cd /Users/hanenshou/Downloads/quant_claude
source venv/bin/activate

# 配置 Tushare Token（编辑 data_collect/config.yaml）
```

### 数据收集

```bash
python main.py init                    # 初始化数据库
python main.py collect-all             # 全量收集所有表（首次建库使用）
python main.py update                  # 增量更新（日常收盘后使用）
python main.py collect-moneyflow-fast  # 并发全量收集资金流向（断点续传）
python main.py collect-report-rc       # 收集券商一致预期数据
python main.py stats                   # 查看数据统计
python main.py refetch-missing         # 检查并修复遗漏数据
```

### 运行回测

```bash
python run_enhanced_fundamental.py
```

输出：`output/csv/enhanced_fundamental_equity.csv` 和 `output/csv/enhanced_fundamental_trades.csv`

## 核心数据表

| 数据表 | 记录数 | 覆盖范围 | 用途 |
|--------|--------|----------|------|
| `daily_price` | 1,110 万 | 2014–2026，全股票（前复权） | 价格/技术分析 |
| `income_statement` | 21.4 万 | 2014–2026，5000+ 只 | F-Score 计算 |
| `balance_sheet` | 20.9 万 | 2014–2026，5000+ 只 | F-Score 计算 |
| `cash_flow` | 21.3 万 | 2014–2026，5000+ 只 | F-Score 计算 |
| `daily_basic` | 1,012 万 | 2016–2026，全覆盖 | 市值/PE/换手率 |
| `moneyflow` | 1,079 万 | 2014–2026，全股票 | 资金流向评分 |
| `index_daily` | 1.5 万 | 2016–2026，6 个指数 | 仓位控制 |
| `report_rc` | 129 万 | 2014–2026 | 券商一致预期 |
| `dividend` | 8.5 万 | 1994–2026 | 红利分析 |
| `stock_basic` | 5,484 行 | 当前列表 | 股票元信息 |

## 增强型基本面策略 v3

### 买入条件（全部满足）

| # | 条件 |
|---|------|
| 1 | 市值 30亿–500亿 |
| 2 | F-Score ≥ 6（Piotroski 9维，同比 YoY，从原始三张报表计算） |
| 3 | ROE > 10% |
| 4 | MA5 > MA20（趋势向上） |
| 5 | RSI(14) < 60（动量未过热） |
| 6 | 量比 > 1.5（成交量放大） |
| 7 | PE 横截面百分位 < 80% |

### 持有条件（任一违反则清仓）

| 条件 | 触发时机 |
|------|----------|
| 最高点回撤 ≥ 7% | 立即，不受最短持有期限制 |
| 价格 < 成本价 × 90% | 立即 |
| MA5 连续 3 日 < MA20 | 持仓满 10 个交易日后检查 |
| F-Score < 6 或 ROE < 10% | 持仓满 10 个交易日后检查 |

### 仓位控制（三档）

| 市场状态 | 条件 | 最大槽位 |
|----------|------|----------|
| 牛市 | 上证 ≥ MA60 | 20 槽（满仓） |
| 震荡 | MA20 ≤ 上证 < MA60 | 10 槽（半仓） |
| 熊市 | 上证 < MA20 | 0 槽（停止新增） |

固定槽位权重 = 1/top_n = 5%，不随市场状态缩放，避免级联再平衡。

### 综合评分（用于 top_n 选股排序）

- **基本面 40%**：F-Score/9 × 33% + min(ROE/30,1) × 45% + (1-PE_pct/100) × 22%
- **技术面 30%**：RSI信号 × 35% + MA强度 × 30% + 价格动量 × 35%
- **资金面 30%**：量比 × 40% + 5日净资金流入排名 × 60%

### 回测结果（2021-02-23 ~ 2026-02-23，5年）

| 指标 | 值 |
|------|----|
| 总收益率 | +68.38% |
| 年化收益率 | 11.45% |
| 最大回撤 | -17.61% |
| 夏普比率 | 0.660 |
| 卡玛比率 | 0.650 |
| 交易次数 | 2,378 |

| 年份 | 收益率 |
|------|--------|
| 2021 | -2.19% |
| 2022 | -14.50% |
| 2023 | +26.43% |
| 2024 | +14.53% |
| 2025 | +24.78% |
| 2026 | +12.63%（截至2月） |

## 项目结构

```
quant_claude/
├── data_collect/              # 数据收集模块
│   ├── config.py             # 配置管理
│   ├── config.yaml           # 配置文件（含 Tushare Token）
│   ├── schema.py             # 数据表结构定义
│   ├── database.py           # DuckDB 管理
│   ├── tushare_api.py        # Tushare API 封装（含限流/重试）
│   ├── collector.py          # 数据收集器（含并发资金流向收集）
│   └── refetcher.py          # 遗漏数据检查与修复
├── policies/                 # 策略模块
│   ├── base_strategy.py      # 策略基类
│   └── enhanced_fundamental_strategy.py  # 增强型基本面策略 v3
├── backtesting/              # 回测引擎
│   ├── vectorized_backtest_engine.py     # 向量化回测引擎
│   └── performance_metrics.py           # 性能指标计算
├── examples/
│   └── basic_analysis.py    # 基础分析示例
├── data/
│   └── quant.duckdb         # DuckDB 数据库（2.9 GB）
├── output/csv/               # 回测输出
├── main.py                   # 数据收集 CLI 统一入口
└── run_enhanced_fundamental.py  # 策略回测入口
```

## 数据更新

### 每日（收盘后）

```bash
python main.py update
```

`update` 自动增量更新：股票列表、日线数据、每日指标（市值/PE）、资金流向、指数日线、券商预测（最近3个月）。

### 季度（财报发布后）

```python
from data_collect.collector import DataCollector
collector = DataCollector()
collector.collect_financial_data(start_date='20240101')  # 财务报表
collector.collect_fina_indicator(start_date='20240101')  # 财务指标
```

### 券商预测数据

```bash
python main.py collect-report-rc                            # 全量收集
python main.py collect-report-rc --incremental             # 增量更新（最近3个月）
```

### 资金流向全量重新收集（5484只，含断点续传）

```bash
python main.py collect-moneyflow-fast                      # 从头开始（断点续传）
python main.py collect-moneyflow-fast --no-resume          # 忽略进度，从头重新收集
python main.py collect-moneyflow-fast --workers 10         # 指定并发线程数
```

## 配置

编辑 `data_collect/config.yaml`：

```yaml
tushare:
  token: "your_tushare_token_here"
  request_interval: 0.2     # API 请求间隔（秒）
  max_retries: 3

database:
  db_path: "data/quant.duckdb"

data_collection:
  start_date: "20140101"
  max_workers: 5            # 并发线程数
```

**Tushare 积分建议**：120分以下 `max_workers=3`，120-2000分 `max_workers=5`，2000分以上 `max_workers=10`。

## 自定义策略

```python
from policies import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data, date):
        # 返回 {ts_code: weight}
        return {"000001.SZ": 0.5, "600000.SH": 0.5}

# 或重写 on_bar 获得每日完全控制权（参考 EnhancedFundamentalStrategy）
```

## 常见问题

**Q: 数据库锁定错误？**
A: 关闭所有正在访问数据库的进程（Jupyter Notebook 等），等待锁自动释放。

**Q: 收集速度慢？**
A: 增加 `max_workers`，减小 `request_interval`（需要更高积分）。

**Q: 为什么某些股票数据少？**
A: 新上市/退市/长期停牌属正常现象，无需修复。

**Q: 如何查看收集进度？**
```python
from data_collect.database import DatabaseManager
db = DatabaseManager()
logs = db.conn.execute("SELECT * FROM update_log ORDER BY created_at DESC LIMIT 10").fetchdf()
print(logs)
db.close()
```

## 注意事项

1. **无投资建议**：本系统仅供研究参考，不构成投资建议
2. **数据存储**：全量数据约需 3 GB 存储空间
3. **前复权价格**：`daily_price` 使用前复权，适合技术分析和回测
4. **财务数据频率**：按季度更新，无需每日收集
5. **数据完整性**：建议每周运行 `python main.py refetch-missing`

---

**版本**: v3 | **最后更新**: 2026-03-03 | **Python**: 3.14 | **数据库**: DuckDB 2.9 GB
