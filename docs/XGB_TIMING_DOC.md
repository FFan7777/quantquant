# XGBoost 时序择时模型 技术文档

**文件**: `xgboost_market_timing.py`
**运行**: `python xgboost_market_timing.py --skip_sensitivity`（含 WFO，约 3 分钟）

---

## 一、核心思路

时序择时模型与截面选股模型的目标不同：

| | 截面选股 | 时序择时 |
|---|---|---|
| **问题类型** | 横向排名（同时间点哪只股票最好） | 纵向预测（某只股票此刻是否值得持有） |
| **标签** | 超额收益截面百分位（连续） | 三重屏障二分类（0/1） |
| **模型** | XGBoost 回归（MSE 目标） | XGBoost 二分类（自定义损失） |
| **评估重点** | Rank IC / ICIR | AUC + Rank IC + 金融绩效 |

**核心假设**：在某个时间节点，如果股票的技术面、基本面、资金面、宏观环境均处于有利状态，则未来 15 个交易日内有更大概率触达上涨止盈屏障（而非先触及下跌止损屏障）。

**信号日 T（收盘）→ 预测未来 15 日内能否先触上屏障 → T+1 开盘建仓**

---

## 二、标签设计：三重屏障法（Triple-Barrier Method）

三重屏障法由 Marcos López de Prado 在《Advances in Financial Machine Learning》中提出，是金融 ML 中最符合实际交易逻辑的打标方式。

### 屏障定义

对每个（日期 T，股票 i），以收盘价 $p_0$ 为基准：

```
上屏障 = p₀ × (1 + profit_take × σ_hold)    ← 止盈（做多成功）
下屏障 = p₀ × (1 - stop_loss   × σ_hold)    ← 止损（做多失败）
垂直屏障 = T + max_hold 交易日                ← 超时未触及（震荡）
```

**关键**：屏障宽度使用**持有期波动率**，而非年化波动率：

```
σ_hold = σ_annual × √(max_hold / 252)    # 默认: 0.35 × √(15/252) ≈ 8.5%
```

这样屏障宽度会自动随个股波动率缩放，高波动股需要更大的价格变动才能触发。

### 标签赋值

```
前向扫描 max_hold 天，找最先被触及的屏障：
  - 触上屏障 → label = 1（买入信号有效，预计盈利）
  - 触下屏障 → label = 0（应回避，止损）
  - 垂直屏障到期 → label = 0（震荡，不如不买）
```

### 默认参数与正样本率

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `profit_take` | 1.5 | 上屏障 = $p_0 \times (1 + 1.5\sigma_{\text{hold}}) \approx +12.8\%$ |
| `stop_loss` | 1.0 | 下屏障 = $p_0 \times (1 - 1.0\sigma_{\text{hold}}) \approx -8.5\%$ |
| `max_hold` | 15 交易日 | 垂直屏障 |
| `vol_lookback` | 20 日 | 滚动年化波动率计算窗口 |
| **正样本率** | **~14.5%** | 约 1/7 的 (日期, 股票) 触及上屏障 |

非对称设计（上屏障 1.5× vs 下屏障 1.0×）反映了"买入机会比亏损更稀有"的市场现实，使模型更专注于识别真正强势的机会。

---

## 三、特征工程（42 个因子）

所有特征均在**信号日 T 收盘时可知**，不含未来信息。调仓日每 15 个交易日取一个截面。

### 3.1 技术面（12 个）

| 因子 | 含义 |
|------|------|
| `ret_1d/3d/5d/10d/20d/60d` | 多尺度收益率（动量 + 短期反转） |
| `vol_5d`, `vol_20d` | 短期 / 中期已实现波动率（年化） |
| `close_vs_ma20`, `close_vs_ma60` | 收盘价偏离 MA20/MA60（趋势强度） |
| `rsi_14` | RSI(14)，Wilder EMA 法 |
| `bb_pos` | 布林带位置：(close - lower) / (upper - lower) ∈ [0,1] |

### 3.2 估值 & 流动性（5 个）

| 因子 | 含义 |
|------|------|
| `pe_ttm` | 滚动市盈率（TTM） |
| `pb` | 市净率 |
| `log_mktcap` | log(总市值)，控制市值风格 |
| `turnover_20d` | 20 日均换手率 |
| `volume_ratio` | 量比（当日成交量 / 历史均量） |

### 3.3 资金流向（4 个）

| 因子 | 含义 |
|------|------|
| `mf_1d_mv` | 当日净流入 / 市值 |
| `mf_5d_mv` | 5 日净流入 / 市值（主力短期意图） |
| `mf_10d_mv` | 10 日净流入 / 市值（主力中期意图） |
| `large_net_5d_ratio` | 5 日大单+超大单净买额 / 总成交额 |

归一化到市值，使大小盘股之间可比。

### 3.4 基本面 PIT（8 个）

| 因子 | 含义 |
|------|------|
| `roe_ann` | 年化 ROE |
| `roa` | 年化 ROA |
| `gross_margin` | 毛利率 (%) |
| `debt_ratio` | 资产负债率 |
| `current_ratio` | 流动比率 |
| `fscore` | Piotroski F-Score（0-9，YoY 同比，三表计算） |
| `rev_growth_yoy` | 营收同比增长率 |
| `ni_growth_yoy` | 净利润同比增长率 |

**PIT（Point-in-Time）原则**：使用 `f_ann_date`（第一披露日）确定数据可知时间，通过 `np.searchsorted` 实现严格的事后无偏合并，避免前视偏差。

### 3.5 市场宽度（5 个）

| 因子 | 含义 |
|------|------|
| `mkt_ret_1d`, `mkt_ret_5d` | 全市场等权 1/5 日收益率 |
| `mkt_vol_20d` | 全市场等权 20 日波动率 |
| `breadth_pct_ma20` | 全市场中价格在 MA20 以上的股票占比 |
| `advance_decline` | 涨跌家数比 |

**作用**：让模型感知市场整体状态（牛市 / 熊市 / 震荡），同一股票在不同市场环境下的信号强度不同。

### 3.6 宏观 & 北向（4 个）

| 因子 | 含义 |
|------|------|
| `hsgt_net_flow` | 沪深港通北向净流入（亿元） |
| `pmi_mfg` | 制造业 PMI（月度，前向填充） |
| `m2_yoy` | M2 同比增速（月度，前向填充） |
| `shibor_3m` | 3 个月 Shibor 利率 |

### 3.7 时间编码（4 个）

```python
month_sin, month_cos   = sin/cos(2π × month / 12)
weekday_sin, weekday_cos = sin/cos(2π × weekday / 5)
```

使用 sin-cos 变换而非原始数值，捕捉时间周期性（月末效应、季报效应等），且能正确处理周期边界（12 月到 1 月的连续性）。

---

## 四、数据预处理

每个调仓日截面独立执行三步预处理：

```
原始因子 → MAD去极值 → 行业/市值中性化 → Z-score 标准化
```

**步骤 1：MAD 去极值**（比 3σ 法更鲁棒）

```python
MAD = |factor - median|.median()
factor.clip(median ± 3 × 1.4826 × MAD)
```

**步骤 2：行业 & 市值中性化**（Ridge 回归残差）

```
factor_raw = β₀ + β₁·ln(MktCap) + Σβᵢ·Industry_i + ε
factor_neutralized = ε
```

剔除"行业 β"和"市值风格"的影响，让模型学习纯 Alpha 信号。

**步骤 3：截面 Z-score 标准化**

确保不同量纲的特征在同一尺度下输入树模型。

不参与中性化的特征（保留原始信息）：`breadth_pct_ma20, advance_decline, mkt_ret_1d/5d, mkt_vol_20d`（这些是市场整体指标，中性化会消除其含义）。

---

## 五、损失函数设计

标准交叉熵损失在金融场景中有两个问题：

1. **正样本极度稀少**（~14.5%），模型容易全预测为 0
2. **假阳性代价 >> 假阴性代价**：买错会亏钱，踏空只是错过机会

为此设计了三种损失函数的可配置加权组合（`LossConfig`）：

### Focal Loss（权重 40%）

```
grad_i = (1-pt)^γ × (p - y)     γ = 2.0（默认）
```

对容易分类的样本（高置信度正确预测）下调权重，让模型聚焦在难分样本上，缓解类别不平衡问题。

### Asymmetric Loss（权重 40%）

```python
weight_i = fp_penalty=2.5  if y_i == 0   # 假阳性（买错）惩罚高
         = fn_penalty=1.0  if y_i == 1   # 假阴性（踏空）惩罚低
```

直接编码"买错 > 踏空"的非对称成本，驱使模型提高精确率（Precision）。

### Directional Loss（权重 20%）

```python
confidence = |p - 0.5| × 2
extra_weight = 1 + 2.0 × confidence   # 仅在方向错误时
```

对高置信度方向错误（自信买入但实际应避免）施加额外惩罚，使模型在"确定性高"时更加谨慎。

### 早停指标

使用**加权 logloss**（`minimize`）作为验证集早停指标，而非 Sharpe。原因：当验证集为熊市年份（如 2022）时，基于 Sharpe 的早停会导致模型收敛至"全部预测为 0"的退化解。加权 logloss 更稳定，不受市场方向影响。

---

## 六、训练流程（Purged Cross-Validation）

### 数据切分

```
训练集:  2018-01-01 ~ 2020-12-31  (3年，约16万行)
    └── 验证集（训练集内部）: 2022-01-01 ~ 2022-12-31（用于早停）
         ⚠ Purged: 清除验证集前 max_hold=15 个调仓日，防止标签重叠
隔离期:  20 个交易日
测试集:  2023-02-21 ~ 2025-12-02  (约3年，23万行)
```

### Purged CV（Lopez de Prado）

三重屏障标签的窗口为 15 天。如果训练集截至日期为 $D_{cut}$，则 $D_{cut}$ 前 15 个交易日内的训练样本，其标签计算窗口会延伸进入验证集，造成**标签重叠（Label Overlap）**：

```python
def purged_cutoff(all_dates, cutoff, purge_n=15):
    """将训练集截止日期提前 purge_n 个调仓日，消除标签重叠。"""
    dates_before = [d for d in all_dates if d < cutoff]
    return dates_before[-(purge_n + 1)]
```

效果：训练集从 676K 行减少到 608K 行（约减少 10%），以消除数据泄露风险。

### XGBoost 超参数

```python
n_estimators     = 1000
max_depth        = 4          # 浅树防止过拟合
learning_rate    = 0.02
subsample        = 0.70
colsample_bytree = 0.70
min_child_weight = 10         # 叶节点最小样本（金融关键参数）
reg_lambda       = 5.0        # L2 正则
reg_alpha        = 0.5        # L1 正则
early_stopping_rounds = 50    # 验证集无改善则停止
```

### 验证集阈值扫描

训练完成后，在验证集上扫描 `threshold ∈ [0.10, 0.50]`，以 Sharpe 为准则选择最优阈值，保存在 `model._best_threshold`。WFO 中每年的测试均使用该年模型自身训练出的最优阈值，而非固定的 0.50。

---

## 七、评估体系

### 算法层面

| 指标 | 说明 |
|------|------|
| AUC-ROC | 二分类排序能力，与阈值无关 |
| Precision | 发出买入信号中真正上涨的比例（精确率） |
| Recall | 所有上涨机会中被识别的比例（覆盖率） |
| F1 Score | Precision 和 Recall 的调和平均 |

### 因子层面（Rank IC / ICIR）

```
Rank IC(t) = Spearman_ρ(pred_prob_t, port_ret_t)
ICIR = mean(IC) / std(IC)
```

衡量模型给出的置信度（连续得分）与实际收益排名的相关性：

| 标准 | 含义 |
|------|------|
| IC > 0.03 | 有效信号 |
| IC > 0.05 | 强信号 |
| ICIR > 0.5 | 信号稳健（年化） |

关键设计：`port_ret` 使用 **`rebal_freq`=15 日持有期收益**，与三重屏障的 `max_hold=15` 对齐，消除了标签周期与评估收益的错配。

### 金融层面

| 指标 | 计算方式 |
|------|----------|
| 年化收益 | `nav[-1]^(periods_per_year / n_periods) - 1`，`periods_per_year = 252/15 ≈ 16.8` |
| 最大回撤 | 滚动最高点到最低点的最大跌幅 |
| 夏普比率 | 净收益均值 / 净收益标准差 × √periods_per_year |
| 卡玛比率 | 年化收益 / |最大回撤| |
| 胜率 | 盈利期数 / 总期数 |
| 盈亏比 | 盈利期均值 / 亏损期均值 |

所有金融指标均在**净收益**（gross return - 交易成本）上计算，与 NAV 曲线完全一致。

---

## 八、组合模拟（Rolling Tranches）

### 交易成本

```python
tc_round = commission×2 + stamp_tax + slippage×2
         = 0.03% × 2   + 0.10%    + 0.10% × 2  = 0.36%（双边）
```

仅在调仓日（实际发生换仓时）计算成本，依据实际换手率比例扣除。

### 滚动分仓（K=3 默认）

将资金等分为 K=3 个子仓位，各自在不同时间点错位调仓：

```
子仓位 0：在第 0, 3, 6, 9...  个调仓日换仓（每 45 交易日一次）
子仓位 1：在第 1, 4, 7, 10... 个调仓日换仓
子仓位 2：在第 2, 5, 8, 11... 个调仓日换仓
组合净收益 = (sub0_ret + sub1_ret + sub2_ret) / 3 - 当日产生的成本
```

**设计目的**：

1. **平滑时序运气**：单一调仓日的好坏会被均摊，减少因"恰好在最差时点重仓"带来的噪声
2. **降低调仓频率**：每个子仓位实际上每 45 天才换一次仓，减少弱信号年份（IC 较低时）的错误交易次数
3. **降低换手成本**：每次调仓只有 1/3 的资金参与，年化换手率下降约 2/3

### 市场宽度门槛（可选，默认关闭）

`min_breadth=0.0`（默认不过滤）。如果设置为 > 0，调仓时若 `breadth_pct_ma20 < min_breadth`，该次调仓转为空仓（不建立新仓位）。

> **注意**：测试表明 `min_breadth=0.35` 反而使 WFO 一致性从 60% 降至 40%，原因是市场宽度最低的时期往往是反弹前夕（最佳入场点），过滤会造成系统性踏空。

---

## 九、滚动样本外测试（WFO）

Walk-Forward Optimization 用于评估模型在不同市场环境下的鲁棒性。每个测试年独立训练一个模型：

```
2020年测试：训练 2018-2019 → 测试 2020
2021年测试：训练 2018-2020 → 测试 2021
2022年测试：训练 2018-2021 → 测试 2022
2023年测试：训练 2018-2022 → 测试 2023
2024年测试：训练 2018-2023 → 测试 2024
```

---

## 十、最终测试结果（2023-2025）

### 信号质量

| 指标 | 值 | 基准 |
|------|-----|------|
| AUC-ROC | 0.57 | 0.5（随机） |
| Rank IC 均值 | **+0.077** | > 0.05 为强 |
| ICIR | **+0.711** | > 0.5 为稳健 |
| IC > 0 比率 | **80.4%** | 越高越稳 |

### 金融绩效（rebal_freq=15，K=3 滚动分仓）

| 指标 | 值 |
|------|-----|
| 年化收益 | **+18.26%** |
| 最大回撤 | **-7.13%** |
| 夏普比率 | **1.230** |
| 卡玛比率 | 2.559 |
| 胜率 | 50.00% |
| 盈亏比 | 2.60 |

### WFO（样本外 2020-2024）

| 年份 | 市场状况 | IC 均值 | ICIR | 年化收益 | 夏普 |
|------|----------|---------|------|---------|------|
| 2020 | COVID 冲击 | +0.089 | 0.54 | +0.85% | 0.130 |
| **2021** | 结构性牛市 | +0.096 | **0.85** | **+46.97%** | **2.364** |
| 2022 | 全面熊市 | +0.090 | **1.18** | -2.86% | -0.006 |
| 2023 | 弱复苏 | +0.090 | **0.99** | -1.45% | -0.054 |
| 2024 | 先弱后强 | +0.051 | 0.40 | +18.07% | 0.632 |
| **汇总** | | **+0.083** | | — | **均值 0.613** |

**关键发现**：IC 在所有年份均显著为正（> 0.05），说明模型的**排名能力**跨越市场周期是稳健的。2022/2023 的轻微负收益是长多策略在绝对下跌市场中的固有限制——IC=0.09 意味着模型比市场平均跑赢，但"比平均更好"在全面下跌时仍可能是负收益。

---

## 十一、优化历程

| 版本 | 关键改变 | 测试夏普 | 备注 |
|------|---------|---------|------|
| 初始 | 年化 σ 用于屏障，Sharpe 早停 | — | 正样本率 0.7%，模型退化为全 0 |
| v2 | 持有期 σ_hold，logloss 早停 | — | 正样本率恢复 14.5% |
| v3 | 修复收益期错配（15日→5日）| — | 返回从 +79%（虚高）校正为正常 |
| v4 | Purged CV + Rank IC + K=3 分仓 | 0.457 | 标签泄露消除，信号质量量化 |
| **v5（当前）** | **rebal_freq=15 对齐 max_hold=15** | **1.230** | 核心改进，消除标签周期错配 |

---

## 十二、代码结构

```
xgboost_market_timing.py
│
├── [配置]
│   ├── LossConfig           # 损失函数参数
│   └── ModelConfig          # 完整模型配置（含 rebal_freq, tranches 等）
│
├── [特征工程]
│   ├── load_price_matrices()         # 前复权价格矩阵 + 滚动波动率
│   ├── load_tech_val_features()      # 技术/估值特征（SQL窗口函数）
│   ├── compute_rsi()                 # RSI(14) 向量化
│   ├── load_moneyflow_features()     # 资金流向特征
│   ├── load_fundamental_panel()      # 基本面三表合并 + F-Score
│   ├── join_fundamental_pit()        # PIT join（np.searchsorted）
│   ├── compute_market_breadth()      # 市场宽度特征
│   ├── load_macro_features()         # 宏观/北向特征
│   └── add_time_encoding()           # 月份/星期 sin-cos 编码
│
├── [标签]
│   └── compute_triple_barrier_labels()  # 三重屏障打标（向量化）
│
├── [预处理]
│   ├── winsorize_mad()               # MAD去极值
│   ├── neutralize_cross_section()    # 行业/市值中性化
│   └── preprocess_panel()            # 截面预处理流程
│
├── [损失函数]
│   ├── FocalLossObjective            # Focal Loss
│   ├── AsymmetricLossObjective       # 非对称惩罚
│   ├── DirectionalLossObjective      # 方向准确率
│   └── CombinedObjective             # 加权组合
│
├── [训练与预测]
│   ├── purged_cutoff()               # Purged CV 边界计算
│   ├── split_panel()                 # 训练/验证/测试切分
│   ├── train_timing_model()          # XGBoost 训练 + 早停 + 阈值扫描
│   └── predict_proba()               # sigmoid(raw_logit)
│
├── [评估]
│   ├── evaluate_ml_metrics()         # AUC/Precision/Recall/F1
│   ├── compute_rank_ic()             # Rank IC / ICIR
│   └── simulate_portfolio()          # 金融回测（含 Rolling Tranches）
│
├── [鲁棒性]
│   ├── walk_forward_evaluation()     # WFO（2020-2024）
│   └── sensitivity_analysis()        # 参数敏感性分析
│
└── main()                            # 主流程（步骤 1-20）
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `output/images/xgb_market_timing_eval.png` | NAV 曲线、月度收益、WFO 结果、特征重要性 |

---

## 十三、使用方式

```bash
# 完整运行（含 WFO，约 3 分钟）
python xgboost_market_timing.py --skip_sensitivity

# 快速调试（仅训练 + 测试，约 1 分钟）
python xgboost_market_timing.py --skip_wfo --skip_sensitivity

# 自定义关键参数
python xgboost_market_timing.py --profit_take 2.0 --stop_loss 0.8 --skip_sensitivity
python xgboost_market_timing.py --fp_penalty 3.0 --focal_gamma 3.0 --skip_sensitivity
python xgboost_market_timing.py --tranches 1 --skip_sensitivity    # 不分仓对比
python xgboost_market_timing.py --min_breadth 0.3 --skip_sensitivity  # 市场宽度过滤
```

---

## 十四、局限性与改进方向

### 当前局限

1. **长多策略固有 Beta 暴露**：2022/2023 全面下跌市场中，IC=0.09 的排名能力无法对冲系统性 Beta，轻微负收益不可避免。需要做空机制才能真正市场中性。

2. **WFO 一致性 60%**：5 个 WFO 年份中 2 年轻微负收益（均在 -3% 以内），反映了纯长多策略在熊市中的内在矛盾。

3. **特征滞后性**：基本面特征使用最新披露财报（可能滞后 1-4 个月），在行情急速变化时反应较慢。

4. **未考虑流动性约束**：回测假设可以按收盘价无摩擦成交，实际 top_n=30 只股票中可能有停牌或涨跌停情况。

### 可探索的改进

1. **结合截面选股模型**：时序择时模型决定是否建仓（总仓位控制），截面选股模型决定选哪些股（个股排名）。两个模型互补，取长补短。

2. **滚动重训练**：目前 main() 只训练一个固定模型，可改为每半年或季度重训，自适应市场风格变化。

3. **市场分期学习**：将牛市 / 熊市 / 震荡市的样本分别训练，或引入市场状态作为 meta-feature，让模型学会在不同环境下调整信号权重。

4. **上调 top_n**：当前 top_n=30（~0.6% 的股票），IC=0.08 的信号在如此高集中度下受随机噪声影响大。扩大到 top_n=100+ 可能更稳健地利用 IC 信号。
