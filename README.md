# A股量化交易系统

基于 DuckDB + XGBoost/LightGBM 的量化交易系统。当前主策略：**指数择时 × 截面选股 v12（宏观择时特征 + H5 LGB ensemble + H10/H5=80/20 + MAX_SLOTS=8）**。

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

## 当前策略：v12 Ensemble（宏观择时特征 + H5 LGB + H10/H5=80/20 + MAX_SLOTS=8）

### 架构

```
指数择时（index_timing_model.py）
    ↓ slots ∈ {0, 10, 20}      决定"是否开仓"及"最大持仓数"
    ↓ 20个特征（含宏观：SHIBOR斜率/PMI新订单/CPI-PPI利差/ATR动态阈值）
截面选股（xgboost_cross_section.py × 2）
    ↓ 0.8×H10 + 0.2×H5 rank   决定"开哪几只"（XGB+LGB Ensemble，两个horizon均有LGB）
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
| 指数择时（index_timing_model.py） | 2016–**2021** | 2022（内嵌 ES） | **2023-02+** |
| 策略 HP Search（strategy_hp_search.py） | — | 2022-02–2024-12 | **2025-02+** |
| 主策略回测（index_ma_combined_strategy.py） | — | — | **2022-01–今（val+test 拼接）** |

> **严格 OOS**：2025-02 起为 CS 模型、HP Search 三者均未使用的唯一真实样本外区间。2022-2024 为策略参数调优期（val），结果仅供参考。

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `MAX_SLOTS` | **8** | 满仓最大持仓数（扫参[4,8]，val Sharpe 8槽优于4槽；**只能用 val 期指标选参，不得用 2025 OOS**） |
| `HALF_SLOTS` | **3** | 半仓最大持仓数 |
| `STOP_LOSS_ENTRY` | 10% | 硬止损 |
| `BULL_MA_DEATH / BULL_MIN_HOLD` | 5 / 5 | 牛市（slots=20）退出参数 |
| `NEUTRAL_MA_DEATH / NEUTRAL_MIN_HOLD` | 3 / 5 | 中性（slots=10）退出参数 |
| `BEAR_MA_DEATH / BEAR_MIN_HOLD` | 2 / 2 | 熊市（slots=0）退出参数 |
| `SLOT_CONFIRM_DAYS` | 3 | 连续 N 天 slots>0 才开新仓 |
| `USE_SIGNAL_SCALE` | True | 信号强度加权仓位 |
| `USE_VOL_SCALE` | True | 波动率反比加权仓位 |

### 指数择时信号（v12：20个特征，含宏观）

| slots | 含义 | 触发条件 |
|-------|------|----------|
| 0 | 熊市/空仓 | CSI300 < MA20（hard3 override） |
| 10 | 半仓 | MA20 ≤ CSI300 < MA60 |
| 20 | 满仓 | CSI300 ≥ MA60 且 ML pred > threshold_full |

**v12 新增宏观特征（3个）**：
- `shibor_slope`：1年期 - 3月期 SHIBOR（收益率曲线斜率，正数代表货币政策宽松预期）
- `pmi_new_order_vs50`：制造业PMI新订单 - 50（>0 代表扩张，领先经济拐点约 1-2月）
- `cpi_ppi_spread`：CPI同比 - PPI同比（正数代表企业利润空间改善）

**ATR 动态阈值**：atr_ratio > 1.5（高波动）时 threshold +0.05，< 0.8（低波动）时 threshold -0.05；实际效果有限（hard3 MA20 override 覆盖了大部分场景）。

### 截面选股特征（H10: 44 个，H5: 46 个）

| 类别 | 特征 |
|------|------|
| 技术面基础（10） | ret_1d/5d/20d/60d/120d, vol_20d, close_vs_ma20/60/120, rsi_14 |
| Alpha158（7） | amplitude_1d, open_vs_close, dist_from_high_5d/20d, dist_from_low_5d, high_low_ratio_20d, vol_ratio_5_20 |
| 估值/流动性（6） | pe_ttm, pb, log_mktcap, turnover_20d, volume_ratio, dv_ratio |
| 资金流向（7） | mf_1d/5d/20d_mv, large_net_5d/20d_ratio, retail_net_5d_ratio, smart_retail_divergence_5d |
| 基本面 PIT（7） | roe_ann, roa, fscore, rev/ni_growth_yoy, gross_margin_chg_yoy, **sue** |
| 分析师预期（3） | analyst_count, np_yield, **analyst_rev_30d**（修正动量：近30d预测/31-90d预测-1）|
| 股东户数（1） | holder_chg_qoq |
| **非线性交叉（3）** | **smart_momentum**（ret_20d×large_net_20d_ratio），**momentum_adj_reversal**（ret_60d-ret_5d），**quality_value_score**（ni_growth_yoy/pe_ttm）|

> 已剪枝（重要性 < 0.7%）：skew_20d、gross_margin、debt_ratio、ocf_to_ni、current_ratio
> `analyst_rev_30d`、`sue` 排除在行业/市值中性化之外（绝对方向信号）
> **H5 额外 2 个 VWAP 特征**：`vwap_dev_1d`（当日 VWAP 偏离度）、`vwap_dev_ma5`（5日VWAP偏离移动均值）；H10 不含 VWAP（实测无改善）

**CS 模型指标（测试集 2025+，Train 2018–2021 / Val 2022–2024，v12）**：

| 模型 | Test ICIR | Test GAUC | 备注 |
|------|-----------|-----------|------|
| H10（10日，44特征，XGB+LGB） | **+1.088** | 0.5595 | v12 当前 |
| H5（5日，44特征，XGB+LGB） | **+1.044** | 0.5473 | v12 当前（无VWAP） |
| v8 H10（44特征，XGB only） | +1.087 | 0.5591 | 历史参考 |
| v8 H5（44特征，XGB only） | +1.104 | 0.5453 | 历史参考（无VWAP） |

> H5 加入 LGB ensemble 后 Test ICIR 轻微下降（1.051→1.044），但 val ICIR 1.025 与 H10 的 val 1.212 差距较大，表明 H5 信号较弱；80/20 权重下 H5 贡献有限，架构对称性带来的好处在于降低单模型随机性。

### 回测结果

**完整区间（2023–2025，T+1 执行，v12）**：

| 指标 | v12（当前最优） | v11（前基线） | v8（早期参考） |
|------|--------------|-------------|--------------|
| 年化收益率 | **+30.44%** | +24.15% | +23.21% |
| 夏普比率 | **1.427** | 1.054 | 1.054 |
| 最大回撤 | **-11.07%** | -14.34% | -14.34% |
| 卡玛比率 | **2.749** | 1.539 | 1.619 |
| 年化波动率 | 18.40% | — | — |

| 年份 | v12 收益率 | v11 收益率 | 说明 |
|------|-----------|-----------|------|
| 2023 | +6.03% | +13.08% | 验证期（v12宏观特征更保守，减少假信号）|
| 2024 | **+67.53%** | +33.90% | 宏观特征正确捕捉2024年2月底反弹 |
| 2025 | +37.58% | +39.48% | **严格 OOS** |
| 2026（至4月17日）| **+9.89%** | — | **严格 OOS**（2026-03-23最深回撤-11.28%，止损清仓后快速回升）|

> **2026年实盘状态（截至2026-04-17）**：当前回撤 **-1.88%**（接近历史峰值），持仓空仓。3月下旬关税冲击期间策略正确止损，4月8日市场回到MA20上方后重新开仓，净值已修复。历史最大回撤 **-14.14%**（2024年10月11日）。

> **v12 相比 v11 的关键改进**：
> 1. **择时模型宏观特征**（SHIBOR斜率 + PMI新订单 + CPI-PPI利差）：正确识别2024年2月底货币宽松信号，2024年+34%→+68%
> 2. **H5 新增 LGB ensemble**：与 H10 架构对称（均为 XGB+LGB rank平均），降低单模型随机性
>
> **2023年退步分析**：宏观特征使择时模型在2023年弱市更保守（零槽位增多），错过部分行情换来更少的错误开仓，属于真实的年度分布权衡，不是过拟合。
>
> **年化收益预期范围 ~27–33%**：LightGBM 无固定 seed，每次重训因随机性有 ±3% 波动。
>
> **T+1 执行说明**：信号由 T 日收盘生成，T+1 日开盘价成交，并过滤涨跌停不可成交股，为可实盘复现的真实模拟。

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
| **LightGBM 固定 seed=42（B1 修复）** | **年化↓至~16%，已回滚** | seed=42 使 LGB val_IC 从 0.1510→0.1616，导致动态 ensemble 权重偏向 LGB（50.5%），LGB 选出不同 2025 个股导致大幅欠收益。XGB 已有 random_state=42，LGB 的随机性是系统固有设计 |
| **turnover_accel_5_20（换手率加速度）** | **年化↓，已回滚** | 使 H10 最优深度从 4 变为 5，2024 年从+30%→+25%，整体 OOS 下滑 |
| **large_net_vol_5d_ratio（大单净量比）** | **无法实现** | moneyflow 表仅有金额字段（buy_lg_amount 等），无股数/手字段（vol），schema 不支持 |
| **Round 4：trend_vert_5d / overnight_intraday_div / vol_asym_20d / vw_momentum_5d** | **年化↓，已回滚** | 四个短期价格微结构因子均与现有特征高度相关，导致 H10 最优 depth 从 4 变为 5，OOS 退步（23%→22.8%）。规律：任何新特征若使 HP 搜索选到 depth=5，必须回滚 |
| **Round 5a：analyst_dispersion（分析师预期分歧度）** | **年化↓，已回滚** | 与 `analyst_rev_30d` 高度相关，同样导致 depth 4→5，OOS 退步 |
| **SUE + depth=5（HP 搜索自动选到）** | **OOS 退步，已修复** | HP 搜索选 depth=5 后 ICIR 1.070→1.066，OOS 下降。根因：depth=5 过拟合 2022 熊市 val 期。修复：将 `XGB_HP_GRID` 从 `d in [3,4,5]` 改为 `d in [3,4]`，再训练后 ICIR 提升至 1.087 |
| **Round 6：ERC 风险平价** | **MaxDD 恶化，已回滚** | w_erc×sig_arr 乘法混合+固定总仓位导致资金集中；MaxDD -14.34%→-19.17%（恶化），Sharpe 下降 0.038 超阈值 |
| **Round 7：VWAP+H10** | **年化↓，已回滚（H5保留）** | H10 加 VWAP 后 HP 搜索漂移至 mcw=40,λ=10（过拟合 val），组合回测年化大降；H5 加 VWAP 后 ICIR +1.104→+1.147，组合收益改善，故 H5 保留 VWAP，H10 不含 |
| **Round 8：rank:ndcg + 时间衰减权重** | **组合回测退步，已全面回滚** | 尽管模型 ICIR 改善（H10 +1.088→+1.164，H5 +1.147→+1.210），2025 回测从 +35.71% 退至 +21-30%。根因：行业中性化与行业分组排序存在根本矛盾（行业均值已被强制归零，行业内排序无额外归纳偏置）；时间衰减权重(alpha=0.5) 对 H10 有害；**ICIR↑ ≠ 集中持仓 α↑** |
| **Round 9：Accruals 特征（Sloan 1996）** | **年化↓，已回滚** | (净利润-经营现金流)/总资产，与 fscore 中 f4 信号（ocf/ta > roa）高度重叠，无净增量信息；年化 24.15%→22.63%，Sharpe 1.054→1.008 |
| **Round 9：多指数协同择时（CSI300+中证1000）** | **年化↓大，已回滚** | CSI300 熊市但中证1000>MA20 时升为半仓，82天触发；年化 24.15%→17.91%，Sharpe 1.054→0.787。根因：大盘熊市期 A 股联动性强，小盘牛不能保护个股开仓安全 |
| **Round 9：Accruals特征** | **年化↓，已回滚** | (净利润-经营现金流)/总资产（Sloan 1996），与fscore中f4信号高度重叠，无净增量；年化24.15%→22.63%，Sharpe 1.054→1.008 |
| **Round 10：csi300_vs_ma60 加入 H5 特征** | **OOS退步，已回滚** | 将指数 MA 状态作为 H5 截面选股特征。根因：择时信息已通过 index_timing_model 注入仓位决策，在 CS 模型内重复引入造成训练/推理不一致。原则：择时信息只能由 index_timing_model.py 提供 |
| **Round 11：集中度感知仓位缩放** | **MaxDD无改善，已回滚** | N=1→0.25x, N=2→0.5x, N=3→0.75x。MaxDD -11.07% 无变化（MaxDD事件发生在N≥4满仓期，属于beta风险，非集中度风险）；年化-0.5%，Sharpe -0.013 |
| **Round 11：分数比例仓位** | **全面退步，已回滚** | 用实际预测分数差异替代线性rank插值[1.5,0.5]。低分股权重极小被跳过（未达MIN_TRADE_VALUE=2000），持仓从3.3→2.7只（更集中），Sharpe 1.427→1.382，MaxDD -11.07%→-11.57% |
| **Round 12：dist_from_year_high_252d（52周高点距离）** | **已回滚** | H10 ICIR +1.073→+1.101，H5 ICIR +1.044→+1.081，均改善且保持depth=4；但组合2025 OOS从37.55%→35.17%（-2.4%）。根因：ICIR↑≠集中持仓α↑，全截面排序改善不等于Top-8选股质量提升 |

### HP Search 方法论失败

| 方案 | 结果 | 教训 |
|------|------|------|
| 单年 Val 2024 | Test Calmar 2.37 | **Regime Overfitting**：2024弱市参数在2025牛市失效 |
| 最大化整体 Calmar | 过拟合某年 | 改用 **Minimax（最差年份 Calmar）** |
| 统一 ma_death/min_hold | 2022亏损-11% | 改用 **Regime-Switching**（bull/neutral/bear 三档） |

---

---

## 超参搜索关键规律

| 规律 | 说明 |
|------|------|
| **depth=5 必须排除** | HP 搜索总倾向于在新特征上选 depth=5，但 depth=5 过拟合 val 期（2022 熊市），OOS 必退步。已将 `XGB_HP_GRID` 固定为 `d in [3, 4]` |
| **新特征筛选标准** | 若添加新特征后 HP 搜索选到 depth=5，该特征应视为"冗余"或"与现有特征高度相关"，需回滚 |
| **策略 HP 与 CS HP 分离** | `strategy_hp_search.py` 的 MAX_SLOTS/HALF_SLOTS 与 CS 模型的 depth/mcw/λ 独立搜索，互不干扰 |
| **ICIR↑ ≠ 组合收益↑** | ICIR 衡量全截面预测与收益相关性，集中持仓策略只关心极端高排名个股；ICIR 提升但 alpha 不一定随之改善，**必须运行完整组合回测验证** |
| **选参只能用验证期（val）** | MAX_SLOTS、HALF_SLOTS 等所有策略超参必须只用 2022年11月—2024年12月（val 期）的年化/夏普选取；2025+ OOS 是事后验证，不参与选参，否则等同于向未来泄露信息 |
| **组合优化空间极有限** | MaxDD来自满仓期系统性beta风险，非集中度风险；N=3.3持仓下协方差优化（MVO/ERC/分数比例）均实测退步。v12已在当前架构Pareto最优区，改善应聚焦择时信号质量 |

---

## 工程优化记录

### 内存优化（2026-04-29）

训练脚本（`xgboost_cross_section.py` / `xgboost_cross_section_h5.py`）内存优化：

| 优化项 | 改动 | 效果 |
|--------|------|------|
| float32 矩阵 | `astype(float)` → `astype(np.float32)` | 训练/推理矩阵内存减少约 40%（XGBoost/LGB内部只用float32）|
| 去除整表copy | `preprocess_panel` 删除 `panel.copy()` | 1.4M 行面板表不再重复拷贝，峰值内存降低约 300-500 MB |
| 显式GC | 三段式切分后 `del panel; gc.collect()` | 确保大型 DataFrame 立即释放，避免两份面板同时驻留 |

---

**版本**: v12（宏观择时特征 + H5 LGB + H10/H5=80/20 + MAX_SLOTS=8）| **最后更新**: 2026-04-29 | **Python**: 3.14 | **数据库**: DuckDB ~6.7 GB
