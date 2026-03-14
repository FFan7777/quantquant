# XGBoost 截面选股模型 技术文档

**文件**: `xgboost_cross_section.py`
**运行**: `python xgboost_cross_section.py`（约 3-4 分钟）

---

## 一、核心思路

截面选股不预测大盘涨跌，而是在同一时间点**横向比较全市场股票**，找出未来 10 个交易日相对表现最强的股票。这是一个排序问题（Learning to Rank）：不需要精确预测收益率数值，只需要排对名次。

**信号日 T（收盘）→ 预测 T+10 的相对表现 → 用于 T+1 开盘建仓**

---

## 二、标签设计

```
label_raw  = stock_ret(T→T+10) - CSI300_ret(T→T+10)   # 个股超额收益
label_norm = cross_sectional_rank(label_raw)             # 截面百分位 [0,1]
```

- 使用**前复权收盘价**计算收益率，避免分红除权断层
- 基准为**沪深 300（000300.SH）**，剥离系统性 Beta
- 对超额收益截面排名归一化，等价于让模型学习相对强弱而非绝对涨跌
- 评估时用**原始超额收益**（未归一化）计算 Rank IC，不受归一化影响

---

## 三、特征工程（26 个因子）

所有特征均在**信号日 T 收盘时可知**，不含未来信息。

### 3.1 技术面（8 个）

| 因子 | 含义 | 数据来源 |
|------|------|----------|
| `ret_1d` | 1 日收益率（短期反转信号） | `daily_price` |
| `ret_5d` | 5 日收益率 | `daily_price` |
| `ret_20d` | 20 日收益率（动量） | `daily_price` |
| `ret_60d` | 60 日收益率（中期动量） | `daily_price` |
| `vol_20d` | 20 日已实现波动率（年化） | `daily_price` |
| `close_vs_ma20` | 收盘价 / MA20 - 1（趋势强度） | `daily_price` |
| `close_vs_ma60` | 收盘价 / MA60 - 1（中期趋势） | `daily_price` |
| `rsi_14` | RSI(14)，Wilder EMA 法，向量化计算 | `daily_price` |

### 3.2 估值 & 流动性（5 个）

| 因子 | 含义 | 数据来源 |
|------|------|----------|
| `pe_ttm` | 滚动市盈率（TTM） | `daily_basic` |
| `pb` | 市净率 | `daily_basic` |
| `log_mktcap` | log(总市值)，用于中性化控制 | `daily_basic` |
| `turnover_20d` | 20 日均换手率（流动性） | `daily_basic` |
| `volume_ratio` | 量比（当日 vs 历史均量） | `daily_basic` |

### 3.3 资金流向（3 个）

| 因子 | 含义 | 数据来源 |
|------|------|----------|
| `mf_1d_mv` | 当日净流入 / 市值 | `moneyflow` |
| `mf_5d_mv` | 5 日净流入 / 市值（聪明钱短期） | `moneyflow` |
| `large_net_5d_ratio` | 5 日大单+超大单净买额 / 总成交额（主力意图） | `moneyflow` |

> 归一化到市值消除了市值大小的影响，使大小盘可比。

### 3.4 基本面（8 个，严格 PIT）

| 因子 | 含义 | 数据来源 |
|------|------|----------|
| `roe_ann` | 年化 ROE（按报告期月数年化） | IS + BS |
| `roa` | 年化 ROA | IS + BS |
| `gross_margin` | 毛利率 (%) | IS |
| `debt_ratio` | 资产负债率 | BS |
| `current_ratio` | 流动比率 | BS |
| `fscore` | Piotroski F-Score（0-9，YoY 同比） | IS + BS + CF |
| `rev_growth_yoy` | 营收同比增长率 | IS |
| `ni_growth_yoy` | 净利润同比增长率（仅上期盈利时计算） | IS |

**PIT（Point-in-Time）原则**：使用 `f_ann_date`（第一披露日）作为数据可知时间。若公司在 T 日尚未披露财报，则使用更早的已披露数据，**严格避免前视偏差**。

实现方式：对每个（股票 × 调仓日），用 `np.searchsorted` 在按 `f_ann_date` 排序的财务序列中找到截至当日的最新披露。

### 3.5 分析师预期（2 个）

| 因子 | 含义 | 数据来源 |
|------|------|----------|
| `analyst_count` | 过去 90 天内发布预测的券商数量（覆盖度） | `report_rc` |
| `np_yield` | 一致预期净利润中位数 / 市值（预期盈利收益率） | `report_rc` + `daily_basic` |

> 覆盖率约 48%（中小盘股通常无券商覆盖），缺失时 XGBoost 内部处理为中性。

---

## 四、数据预处理（每个截面独立执行）

每个调仓日的截面数据经过三步处理：

```
原始因子  →  MAD去极值  →  行业/市值中性化  →  Z-score标准化
```

### 步骤 1：MAD 去极值

使用绝对中位差（MAD）法，比 3σ 法对异常值更鲁棒：

```python
median = factor.median()
MAD = |factor - median|.median()
upper = median + 3 × 1.4826 × MAD
lower = median - 3 × 1.4826 × MAD
factor = factor.clip(lower, upper)
```

### 步骤 2：行业 & 市值中性化

将每个因子对行业哑变量和 log(市值) 做 Ridge 回归，取残差：

```
factor_raw = β₀ + β₁·ln(MktCap) + Σβᵢ·Industry_i + ε
factor_neutralized = ε  (残差)
```

**目的**：剔除"选择某个行业"或"偏向小盘"的系统性风格暴露，让模型学习的是**同等市值、同行业内**的相对优秀度（纯 Alpha）。

> 不中性化的话，模型可能会学到"买银行 = 低 PE = 高分"，这并非真正的选股能力。

### 步骤 3：截面 Z-score 标准化

```python
factor = (factor - mean) / std
```

让所有因子在同一量纲下输入模型，避免量纲差异影响树的分裂。

---

## 五、模型训练

### 样本构造

- **调仓频率**：每 5 个交易日取一个截面，共 389 个截面
- **每截面样本数**：约 4,000-5,000 只股票
- **总样本量**：约 160 万行

由于连续截面的标签存在 **5 天重叠**（10 日标签、5 日间隔），训练集内部有自相关，但已用 Purged Split 处理训练/测试边界。

### XGBoost 超参数

```python
objective        = "reg:squarederror"  # MSE on rank-normalized labels ≈ 排序模型
n_estimators     = 1000
max_depth        = 4                   # 浅树，防止过拟合
learning_rate    = 0.02
subsample        = 0.7                 # 行采样
colsample_bytree = 0.7                 # 列采样
min_child_weight = 30                  # 叶节点最小样本数（金融数据关键参数）
reg_lambda       = 10.0               # L2 正则（强正则化）
reg_alpha        = 0.5                 # L1 正则（稀疏化）
```

`min_child_weight=30` 是金融数据中的关键超参：防止模型在少数"特殊"股票上过拟合，因为金融数据中极端案例（爆雷股、妖股）噪声极大。

### 训练 / 验证 / 测试切分（Purged Split）

```
训练集:  2018-01-01 ~ 2021-12-31  (4年，约 67万行)
验证集:  2022-01-01 ~ 2022-12-31  (1年，约 22万行，用于 Early Stopping)
-------- 隔离期: 20个交易日 --------
测试集:  2023-02-07 ~ 2025-12-16  (3年，约 70万行)
```

**为什么需要隔离期（Embargo）**：标签窗口为 10 日，如果训练集截至 2022-12-31，那么 2022-12-31 之后约 10 个交易日的标签与训练集有重叠。隔离期确保测试集不包含任何与训练集标签相交的样本，消除**时间泄露**。

---

## 六、评估指标

### Rank IC（Information Coefficient）

```
Rank IC(t) = Spearman_ρ(model_score_t, actual_excess_return_t)
```

衡量模型预测的**排名**与真实收益排名的相关性。值域 [-1, 1]，越大越好。

- 业界标准：均值 > 0.03 为有效因子，> 0.05 为优秀
- 本模型测试集：**+0.113**（极优）

### ICIR（IC Information Ratio）

```
ICIR = mean(IC) / std(IC)
```

IC 的"夏普比率"，衡量预测能力的**稳定性**。ICIR > 0.5 为强信号，> 1.0 为极强。

- 本模型测试集：**+1.00**

### GAUC（Group AUC）

对每个截面，计算 AUC（模型能否区分"跑赢大盘的股票"与"跑输大盘的股票"），再按截面样本数加权平均：

```
GAUC = Σ(AUC_t × N_t) / Σ(N_t)
```

- 基准值：0.5（随机）
- 本模型测试集：**0.552**
- 与推荐系统的 GAUC 思路相同：在每个"组"（交易日）内评估排序质量，消除"大盘效应"对评估的污染

### 多空组合收益

每个截面预测得分最高的 Top 10%（做多）与最低的 Bottom 10%（做空）的超额收益之差，验证因子实际可用性。

- 本模型测试集多空均值：**+1.97% / 10日**

---

## 七、结果

| 指标 | 训练集 (2018-2022) | 测试集 (2023-2025) |
|------|:-------------------:|:-------------------:|
| 截面数量 | 243 | 140 |
| Rank IC 均值 | +0.141 | **+0.113** |
| IC 正比率 | 93.8% | 83.6% |
| ICIR | +1.409 | **+0.997** |
| GAUC | 0.556 | **0.552** |
| 多空均值收益 (10日) | 2.87% | 1.97% |
| 多空年化夏普 | 6.82 | 3.80 |

训练集与测试集的指标差距较小，说明**泛化能力良好**，未出现严重过拟合。

---

## 八、代码结构

```
xgboost_cross_section.py
│
├── load_price_pivot()              # 加载前复权收盘价矩阵
├── load_tech_basic_features()      # 技术/估值特征（SQL窗口函数）
├── compute_rsi()                   # RSI(14) 向量化计算
├── load_moneyflow_features()       # 资金流向特征（SQL窗口函数）
├── load_fundamental_panel()        # 基本面PIT面板（三表合并 + F-Score）
├── join_fundamental_pit()          # PIT join（np.searchsorted）
├── load_analyst_features()         # 分析师预期特征
├── compute_labels()                # 10日超额收益标签
│
├── winsorize_mad()                 # MAD去极值
├── neutralize_cross_section()      # 行业/市值中性化（Ridge残差）
├── preprocess_panel()              # 截面预处理流程
│
├── split_purged()                  # Purged时序切分
├── train_xgb()                     # XGBoost训练（Early Stopping）
│
├── compute_rank_ic()               # Rank IC序列
├── compute_gauc()                  # GAUC
├── evaluate()                      # 综合评估
│
└── main()                          # 主流程（~3-4分钟）
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `output/images/xgb_cross_section_eval.png` | IC序列图、累积IC图、多空净值图、特征重要性图 |
| `output/csv/xgb_cross_section_predictions.csv` | 测试集每只股票每个截面的预测分与实际超额收益 |
| `output/csv/xgb_feature_importance.csv` | 26个因子的重要性排名 |

---

## 九、局限性与改进方向

**当前局限**：
1. **标签重叠**：每5日调仓但10日标签，训练集内相邻截面有5日重叠，引入序列自相关
2. **分析师覆盖率低**：`analyst_count` / `np_yield` 仅覆盖约 48% 样本，主要是大中盘股
3. **静态超参**：未做时序交叉验证调参，单一验证集可能带有运气成分
4. **未考虑 ST / 停牌**：回测时涨跌停股票实际无法交易

**可探索的改进**：
- 增加动量因子的更多窗口（3日、10日、120日）
- 增加财务质量因子（应计利润、经营杠杆）
- 使用 `rank:ndcg` 目标函数（直接优化排序），理论上更适合此问题
- 与时序择时模型结合：时序模型决定总仓位，截面模型决定选哪些股
- 滚动重训练（每季度重训，动态捕捉市场风格变化）
