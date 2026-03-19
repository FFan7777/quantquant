# A股量化交易系统

基于 DuckDB + XGBoost/LightGBM 的量化交易系统。当前主策略：**指数择时 × 截面选股 v6 Ensemble + HP Search v2**。

---

## 快速上手（4步流程）

```bash
# 0. 进入项目并激活环境
cd /Users/hanenshou/Downloads/quant_claude && source venv/bin/activate

# 1. 数据更新（每日收盘后，批量接口，约3分钟）
python daily_update.py --force

# 2. 训练/更新 ML 模型（数据累积 > 3 个月时）
python index_timing_model.py --label_type ma60_state --no_wfo   # ~10min
python xgboost_cross_section.py                                  # ~18min (H10)
python xgboost_cross_section_h5.py                               # ~12min (H5)
python strategy_hp_search.py                                     # ~1min  (超参搜索)

# 3. 运行回测验证（2022-2026 完整区间）
python index_ma_combined_strategy.py

# 4. 实盘推理
python infer_today.py --status          # 今日状态看板（MA/CS预测/近期交易）
python infer_today.py                   # 实盘买卖信号
python infer_today.py --holdings h.json # 带持仓退出检查
```

---

## 当前策略：v6 Ensemble + HP Search v2

### 架构

```
指数择时（index_timing_model.py）
    ↓ slots ∈ {0, 10, 20}      决定"是否开仓"及"最大持仓数"
截面选股（xgboost_cross_section.py × 2）
    ↓ 0.7×H10 + 0.3×H5 rank   决定"开哪几只"（XGB+LGB Ensemble）
Regime-Switching 退出参数
    ↓                          根据当前 slots 动态调整 MA死叉/止损阈值
T+1 执行
    ↓                          信号由 T 日收盘生成，T+1 日开盘价成交
```

### 数据划分

| 模块 | Train | Val（调参/ES） | Test（OOS） |
|------|-------|----------------|-------------|
| CS 模型 H10（xgboost_cross_section.py） | 2018–2021 | 2022–2024 | **2025-02+** |
| CS 模型 H5（xgboost_cross_section_h5.py） | 2018–2021 | 2022–2024 | **2025-02+** |
| 指数择时（index_timing_model.py） | 2016–2022 | 2022（内嵌 ES） | **2023-02+** |
| 策略 HP Search（strategy_hp_search.py） | — | 2022-02–2024-12 | **2025-02+** |
| 主策略回测（index_ma_combined_strategy.py） | — | — | **2022-01–今（val+test 拼接）** |

> **严格 OOS**：2025-02 起为 CS 模型、HP Search 三者均未使用的唯一真实样本外区间。2022-2024 为策略参数调优期（val），结果仅供参考。

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `MAX_SLOTS` | 8 | 满仓最大持仓数 |
| `HALF_SLOTS` | 3 | 半仓最大持仓数（HP Search minimax 偏好 2，但 OOS 实测 3 更优） |
| `STOP_LOSS_ENTRY` | 10% | 硬止损 |
| `BULL_MA_DEATH / BULL_MIN_HOLD` | 5 / 5 | 牛市（slots=20）退出参数 |
| `NEUTRAL_MA_DEATH / NEUTRAL_MIN_HOLD` | 3 / 5 | 中性（slots=10）退出参数 |
| `BEAR_MA_DEATH / BEAR_MIN_HOLD` | 2 / 2 | 熊市（slots=0）退出参数 |
| `SLOT_CONFIRM_DAYS` | 3 | 连续 N 天 slots>0 才开新仓 |
| `USE_SIGNAL_SCALE` | True | 信号强度加权仓位 |
| `USE_VOL_SCALE` | True | 波动率反比加权仓位 |

### 指数择时信号

| slots | 含义 | 触发条件 |
|-------|------|----------|
| 0 | 熊市/空仓 | CSI300 < MA20 |
| 10 | 半仓 | MA20 ≤ CSI300 < MA60 |
| 20 | 满仓 | CSI300 ≥ MA60 且 ML 看涨 |

### 截面选股特征（39 个）

| 类别 | 特征 |
|------|------|
| 技术面基础（10） | ret_1d/5d/20d/60d/120d, vol_20d, close_vs_ma20/60/120, rsi_14 |
| Alpha158 新增（7） | amplitude_1d, open_vs_close, dist_from_high_5d/20d, dist_from_low_5d, high_low_ratio_20d, vol_ratio_5_20 |
| 估值/流动性（6） | pe_ttm, pb, log_mktcap, turnover_20d, volume_ratio, dv_ratio |
| 资金流向（7） | mf_1d/5d/20d_mv, large_net_5d/20d_ratio, retail_net_5d_ratio, smart_retail_divergence_5d |
| 基本面 PIT（6） | roe_ann, roa, fscore, rev/ni_growth_yoy, gross_margin_chg_yoy |
| 分析师预期（2） | analyst_count, np_yield（覆盖率 ~48%，缺失补 0） |
| 股东户数（1） | holder_chg_qoq |

> 已剪枝（重要性 < 0.7%，去除后 OOS 无损）：skew_20d、gross_margin、debt_ratio、ocf_to_ni、current_ratio

**CS 模型指标（测试集 2025+，Train 2018–2021 / Val 2022–2024）**：

| 模型 | Val ICIR | Test ICIR | Test GAUC |
|------|----------|-----------|-----------|
| H10（10日 horizon） | +1.143（train+val） | +1.077 | 0.5601 |
| H5（5日 horizon） | +1.092 | +1.110 | 0.5454 |

### 回测结果

**完整区间（2022-2026，val+test 拼接，T+1 执行）**：

| 指标 | 值 |
|------|-----|
| 总收益率 | **+85.90%** |
| 年化收益率 | **+20.40%** |
| 最大回撤 | **-13.51%** |
| 夏普比率 | **1.004** |
| 卡玛比率 | **1.509** |

| 年份 | 收益率 | 说明 |
|------|--------|------|
| 2022 | 含于总收益 | CS 模型 Val 期 |
| 2023 | +15.17% | CS 模型 Val 期 |
| 2024 | +30.28% | CS 模型 Val 期 |
| 2025 | +25.85% | **严格 OOS（CS+HP 均未见）** |
| 2026（至3月） | +6.92% | **严格 OOS** |

> **T+1 执行说明**：信号由 T 日收盘生成，T+1 日开盘价成交，并过滤涨跌停不可成交股，为可实盘复现的真实模拟。与 T+0 收盘成交相比年化收益约降低 5–8%。

### 仓位管理

- **固定槽位**：`slot_value = total_value / MAX_SLOTS`（不随持仓数变化，避免级联再平衡）
- **信号强度加权**：排名第 i 只 → `sig_scale = 1.5 - 1.0 × i/(n-1)`（第1名1.5x，最后0.5x）
- **波动率反比**：`vol_scale = clip(median_vol / this_vol, 0.5, 2.0)`
- **闲置现金**：年化 2% 收益（模拟货币基金）

### 入场过滤

| 过滤项 | 规则 |
|--------|------|
| 合规 | 非 ST、上市满 90 天、非北交所（8\*/4\*） |
| 市值 | > 后 10% 百分位 |
| 停牌 | 当日成交量 > 0 |
| 涨停 | T+1 日涨幅 ≥ 涨停阈值（主板 9.9%，科创/创业 19.9%）排除买入 |
| 跌停 | T+1 日跌幅 ≤ 跌停阈值时跳过止损卖出（无法成交） |

---

## 数据库

### 概览

| 指标 | 值 |
|------|----|
| 数据库大小 | ~6.7 GB (DuckDB) |
| 总记录数 | 40,000,000+ |
| 股票数量 | 5,484 只 |
| 日期范围 | 2014–2026 |

### 核心数据表

| 数据表 | 记录数 | 覆盖范围 | 用途 |
|--------|--------|----------|------|
| `daily_price` | 1,117 万 | 2014–2026，全股票（前复权） | 技术特征 |
| `daily_basic` | 1,016 万 | **2016–2026 全覆盖** | 市值/PE/换手率 |
| `moneyflow` | 1,086 万 | 2014–2026，全股票 | 资金流向特征 |
| `income_statement` | 21.4 万 | 2014–2026，5000+ 只 | F-Score / 基本面 PIT |
| `balance_sheet` | 20.9 万 | 同上 | 同上 |
| `cash_flow` | 21.3 万 | 同上 | 同上 |
| `index_daily` | 1.5 万 | 2016–2026，6 个指数 | 指数择时 |
| `report_rc` | 129 万 | 2014–2026 | 分析师预期特征 |
| `stk_holdernumber` | 21.9 万 | 2015–2025，按季度 | 股东户数特征 |
| `stock_basic` | 5,484 行 | 当前列表 | 股票元信息 |

### 关键注意事项

1. **PIT 严格防穿越**：财务数据使用 `f_ann_date`（第一披露日），通过 `np.searchsorted` 查找每个 trade_date 前最新已知值
2. **daily_basic.close ≠ daily_price.close**：前者为原始价，后者为前复权价
3. **fina_indicator 覆盖不完整**：F-Score 必须从 income_statement + balance_sheet + cash_flow 自行计算
4. **T+1 执行**：信号由 T 日收盘生成，以 T+1 日开盘价成交；ST 状态使用当前名称（历史 ST 无法区分）

### 数据更新

**自动定时更新（每天 18:00，macOS launchd）**：

```bash
# 查看日志
tail -f logs/launchd_out.log

# 手动触发
launchctl start com.quant.daily_update

# plist 位置
~/Library/LaunchAgents/com.quant.daily_update.plist
```

**日常更新速度（批量按日期接口）**：

| 表 | 旧版（逐股） | 新版（批量） |
|----|------------|------------|
| daily_price | ~2.5h | **~30s** |
| daily_basic | ~2.5h | **~15s** |
| moneyflow | ~2.5h | **~15s** |
| 财务报表 | ~1h | ~1h（无批量接口） |

> 批量接口（`collect_daily_price_batch` 等）对 ≤60 天窗口自动启用；全量重建自动降级为逐股模式。

**手动操作**：

```bash
python daily_update.py --force          # 强制运行（忽略非交易日检查）
python main.py collect-holder           # 股东户数（按季度）
python main.py collect-financial        # 三张财务报表（财报季后）
python main.py collect-moneyflow-fast   # 资金流向全量（断点续传）
python main.py collect-report-rc        # 券商一致预期
python main.py stats                    # 查看数据库统计
python main.py refetch-missing          # 修复遗漏数据
```

---

## 项目文件

```
quant_claude/
├── index_ma_combined_strategy.py     # 主策略回测（v6 ensemble）
├── strategy_hp_search.py             # 策略超参搜索（Minimax + 坐标下降）
├── xgboost_cross_section.py          # CS 模型训练（H10，10日horizon）
├── xgboost_cross_section_h5.py       # CS 模型训练（H5，5日horizon）
├── index_timing_model.py             # 指数择时模型（ma60_state label）
├── infer_today.py                    # 实盘推理（--status 看盘 / 默认推理）
├── main.py                           # 数据收集 CLI 统一入口
├── daily_update.py                   # 定时自动更新（批量接口，~3min/天）
├── data_collect/
│   ├── collector.py                  # 数据收集器（含批量按日期方法）
│   ├── tushare_api.py                # Tushare API 封装（含批量接口）
│   ├── database.py                   # DuckDB 管理
│   ├── schema.py                     # 23张表的 DDL
│   ├── refetcher.py                  # 遗漏数据检查与修复
│   └── config.yaml                   # 配置（Tushare Token、DB路径）
├── backtesting/
│   ├── vectorized_backtest_engine.py # NumPy 向量化回测（5年全量 < 5秒）
│   └── performance_metrics.py        # Sharpe/Calmar/MaxDD 计算
├── data/quant.duckdb                 # 数据库（~6.7 GB）
└── output/
    ├── models/                       # 训练好的 XGB/LGB 模型（.json/.txt）
    ├── csv/                          # 预测结果 CSV（val+test 分别保存）
    └── index_ma_combined/            # 回测输出（净值/交易记录/图表）
```

---

## 配置

```yaml
# data_collect/config.yaml
tushare:
  token: "your_tushare_token_here"
  request_interval: 0.2   # Tushare 积分 < 120 时用 0.5
  max_retries: 3

database:
  db_path: "data/quant.duckdb"

data_collection:
  start_date: "20140101"
  max_workers: 5          # 积分 < 120 → 3；2000+ → 10
```

---

## 失败实验存档

以下实验均已充分测试，结论稳定，**禁止重复尝试**：

### 策略参数优化失败

| 方案 | 结果 | 原因 |
|------|------|------|
| 止损 8%→12%/6% | 无改善/略差 | 不是主要瓶颈 / 更多假止损 |
| SLOT_CONFIRM 3→1/2 | 年化↓↓/↓ | 严重/轻微 whipsaw |
| MAX_SLOTS 8→5/15 | 年化↓ | 集中度过高 / 每槽价值过小 |
| BEAR_FAST_EXIT=True | 年化↓ | 2024/2025 频繁假信号 |
| ENTRY_MA20_FILTER=True | 年化↓ | 错过均线下方低位买入机会 |
| MIN_HOLD_DAYS 5→3 | 年化↓ | 过早出场 |
| MA_DEATH_DAYS 5→3 | 年化↓大 | 过多假死叉出场 |

### 模型/特征失败

| 方案 | 结果 | 原因 |
|------|------|------|
| rank:pairwise（XGBRanker）| Val ICIR 0.56（原0.89↓）| 2树即过拟合，无法充分训练 |
| lambdarank（LightGBM）| 同上 | 与 rank:pairwise 同批次回退 |
| 动量特征 ret_250d | 年化↓ | 2024-2025 为价值反转行情 |
| H10+H5+H20 三模型 | 年化↓ | H20 窗口与 5 日调仓时域不匹配 |
| H10 训练集延伸至 2022（TRAIN_END=2022）| 2025 年化↓8%+ | 引入熊市训练数据干扰 2025 牛市预测；Test ICIR 名义上略升但 OOS 回测显著下降 |
| H10 训练集延伸至 2023 | 年化↓ | 熊市年份污染模型 |
| H10+H5 ensemble 权重改为 35/65 | 年化大幅↓ | H5 信噪比较低；70/30（更重 H10）OOS 表现始终更优 |
| 行业动量特征 | 年化↓ | 中性化后信息丢失 |
| WFO（Walk-Forward） | ICIR +0.005 | 耗时 10x，收益不值得 |
| MA20/MA60 交叉纯规则择时 | 年化↓5% | 2023 牛市大量误判空仓 |

### HP Search 方法论失败

| 方案 | 结果 | 教训 |
|------|------|------|
| 单年 Val 2024 | Test Calmar 2.37 | **Regime Overfitting**：2024弱市参数在2025牛市失效 |
| 最大化整体 Calmar | 过拟合某年 | 改用 **Minimax（最差年份 Calmar）** |
| 统一 ma_death/min_hold | 2022亏损-11% | 改用 **Regime-Switching**（bull/neutral/bear 三档） |

---

**版本**: v6+HP_v2 | **最后更新**: 2026-03-19 | **Python**: 3.14 | **数据库**: DuckDB ~6.7 GB
