# 策略级超参数搜索 技术文档

**文件**: `strategy_hp_search.py`
**运行**: `python strategy_hp_search.py`（约 30 秒）
**最后更新**: 2026-03-15

---

## 一、背景与动机

### 问题

在构建联合策略（指数择时 + 截面选股）时，存在两个层次的超参数：

| 层次 | 参数 | 原有调优方式 |
|------|------|------------|
| **模型层**（截面选股） | XGBoost/LGB 树深、学习率、特征数等 | ✅ 已用 Train/Val/Test 三段划分（`xgboost_cross_section.py`）|
| **策略层** | 槽位数、止损幅度、MA死叉天数等 | ❌ 通过人工观察 2021-2026 完整回测调整，存在 test-set leakage |

策略层参数（`index_ma_combined_strategy.py` 中的全局常量）之前是靠"看完整回测结果调参"确定的。这本质上是在用 test 数据做调优，属于数据泄漏。

### 解决方案

本脚本将同样的三段划分应用于策略层超参数搜索：

```
Train（模型训练）: 2018-01 ~ 2023-12
Val  （策略HP搜索）: 2024-02 ~ 2024-12   ← 只在此期间搜索策略参数
Test （唯一OOS评估）: 2025-02 ~ 2026-03  ← 完成后不再修改参数
```

---

## 二、搜索参数与范围

### 被搜索的参数

| 参数 | 含义 | 搜索范围 | 默认值 |
|------|------|---------|--------|
| `max_slots` | 满仓最大持股数 | 4, 6, 8, 10 | 8 |
| `half_slots` | 半仓最大持股数 | 2, 3, 4, 5 | 3 |
| `stop_loss` | 入场价回撤止损幅度 | 4%~12% | 8% |
| `ma_death_days` | MA5<MA20 连续 N 天触发死叉退出 | 2~7 | 5 |
| `min_hold_days` | 触发MA死叉检查的最短持有期 | 2~10 | 5 |
| `mktcap_pct_cut` | 排除市值后 X% 微小盘股 | 5, 10, 15, 20 | 10 |

### 不搜索的参数（保持默认）

- `trailing_stop=1.0`（追踪止损禁用，依赖MA死叉 + 硬止损）
- `slot_confirm_days=3`（连续3天slots>0才开仓，防whipsaw）
- `use_vol_scale=True`，`use_signal_scale=True`（已在模型层验证有效）
- `min_listed_days=90`（合规过滤，非alpha相关）

---

## 三、搜索方法：坐标下降（Coordinate Descent）

每次固定其他维度，对当前维度的所有候选值进行回测，取最优后进入下一维度。

```
Step A: 搜索 (max_slots, half_slots) → 基于 DEFAULT_PARAMS
Step B: 搜索 stop_loss               → 固定 Step A 最优结果
Step C: 搜索 (ma_death_days, min_hold_days) → 固定 Step B 最优结果
Step D: 搜索 mktcap_pct_cut          → 固定 Step C 最优结果
```

**优化目标**：Calmar 比率（= 年化收益 / 最大回撤绝对值），要求年化收益 > 0。

**总回测次数**：~30 次（val 期，每次约 0.8 秒）

---

## 四、数据依赖

```
strategy_hp_search.py
├── output/csv/xgb_cs_pred_val.csv         ← H10 train-only模型在val期的预测
│                                              (由 xgboost_cross_section.py 生成，Step 13改动)
├── output/csv/xgb_cross_section_predictions.csv  ← H10 final模型在test期的预测
└── output/csv/index_timing_predictions.csv       ← 指数择时模型信号（含val+test期）
```

### xgb_cs_pred_val.csv 的生成方式

`xgboost_cross_section.py` 的 Step 13 改为两步返回：

1. **Stage 1**：在 train 上训 XGB/LGB（val 为 early stopping），得到 `model_train_only`
2. 用 `model_train_only` 对 val 期数据推理，保存为 `xgb_cs_pred_val.csv`（**严格 PIT，无未来信息**）
3. **Stage 2**：在 train+val 合并后重训，得到 `model_final`（用于 test 推理）

这确保 val 期预测来自"当时可知的模型"，与实盘部署逻辑一致。

---

## 五、实验结果（2026-03-15）

### 5.1 搜索过程（Val 2024）

| Step | 最优组合 | Val Calmar |
|------|---------|-----------|
| A: slots | ms=4, hs=2 | 2.18 |
| B: stop_loss | sl=8%（不变） | 2.18 |
| C: MA退出 | md=3, mh=3 | 2.93 |
| D: 市值过滤 | mc=5（无变化） | 2.93 |

**最终 Val 最优参数**：`max_slots=4, half_slots=2, stop_loss=8%, ma_death_days=3, min_hold_days=3`

### 5.2 Val vs Test 绩效对比

| 参数组 | Val 2024 年化 | Val Calmar | Test 2025+ 年化 | Test Calmar |
|--------|-------------|-----------|----------------|------------|
| Val 调优参数（ms=4,hs=2,md=3,mh=3） | **+30.8%** | **2.93** | +26.3% | 2.37 |
| 默认参数（ms=8,hs=3,md=5,mh=5） | +20.6% | 1.64 | **+47.5%** | **4.38** |

---

## 六、关键发现

### 6.1 mktcap_pct_cut 无效

市值过滤（5%/10%/15%/20%）对 Calmar 完全没有影响，原因是过滤掉的股票本身在 CS 模型评分中排名就很低，不会被选入持仓。**可以简化：固定 10% 即可，无需搜索。**

### 6.2 Val 调优参数在 Test 上反而更差（Regime Overfitting）

**这是核心发现**。Val 期（2024）偏向震荡/弱势市，最优策略是：
- 少持股（ms=4）→ 降低市场敞口
- 快退出（md=3, mh=3）→ 尽快止损，避免持仓拖累

但 Test 期（2025）是强牛市，这套参数造成：
- 满仓槽位只有 4 个，错过大量多头机会
- 3天就判定死叉，频繁换仓，追高杀跌

结论：**用单一年度 val 数据做策略 HP 搜索，与 val 期和 test 期 regime 是否一致高度相关**。这是数据量不足导致的泛化失败，不是方法本身的错误。

### 6.3 参数稳健性（敏感性测试结果）

| 参数 | 稳健性 | Calmar 范围 | 建议 |
|------|--------|-----------|------|
| `half_slots` | ✓ 高度稳健 | 0.04 | 可放心调 |
| `stop_loss` (5%~12%) | △ 中等 | 0.68 | 5%~10% 均可 |
| `max_slots` | △ 中等 | 0.76 | ≤6 时明显更好（val 2024） |
| `ma_death_days` | ⚠ 高度敏感 | 1.61 | 强 Regime 依赖，不应靠单年 val 调 |
| `min_hold_days` | ⚠ 高度敏感 | 0.78 | 同上 |

---

## 七、正确做法：多 Regime Val 期

### 当前限制

| 限制 | 原因 |
|------|------|
| Val 只有 1 年（2024） | XGBoost 截面模型的 Val 期定在 2024 |
| 2024 market regime：震荡/弱势后强反弹 | 不代表 2025 强牛市，也不代表 2022 熊市 |

### 更好的策略层 HP 搜索方案

**方案 A：扩大 Val 期（推荐）**

将截面模型和策略的训练/验证/测试重新划分：
```
截面模型 Train: 2018-2021
策略 Val（HP搜索）: 2022-2023（含熊市+复苏，两种极端 regime）
Test（OOS）: 2024-2026
```
代价：需要重训截面模型，test 期缩短为 2 年。

**方案 B：Robustness Criterion（不改变现有划分）**

在 Val 期内按年度分段，以最差子期绩效作为优化目标：

```python
# 优化目标改为：min(各子期 Calmar) 而非 全期 Calmar
score = min(calmar_2022, calmar_2023, calmar_2024)
```

这强制选择对所有子 regime 都稳健的参数，而非拟合特定市场环境。

**方案 C：参数固化（当前可行方案）**

对于高度敏感（Regime 依赖）的参数（`ma_death_days`, `min_hold_days`），**不通过 HP 搜索确定，而是基于领域知识固化**：

- `ma_death_days=5`：需要连续5天死叉才退出，过滤假信号（源于 enhanced_fundamental_strategy v3 的经验）
- `min_hold_days=5`：至少持有5天，避免当日涨跌触发无效退出

对于低敏感度参数（`stop_loss`, `half_slots`），单年 val 搜索结果可信度更高。

---

## 八、与截面选股 HP 搜索的对比

| | 截面选股 HP 搜索 | 策略 HP 搜索（本文） |
|---|---|---|
| 优化目标 | Val 期 Rank IC 均值 | Val 期 Calmar 比率 |
| Val 数据量 | 44个截面 × 5000只股票 ≈ 22万样本 | 1年×约240日回测 |
| 每次评估耗时 | ~5秒 | ~0.8秒 |
| 搜索方法 | Bayesian HPO（Optuna） | 坐标下降 |
| Regime 泛化 | IC 是截面统计量，对 regime 不敏感 | Calmar 直接反映 regime |
| 泛化可靠性 | ✅ 高 | △ 中（取决于 val 期长度） |

**原因**：截面 Rank IC 是每个调仓截面内部的相对排序指标，与市场整体涨跌方向无关。策略 Calmar 则直接受 regime 影响——同样的参数在熊市和牛市下表现天壤之别。

---

## 九、当前建议

1. **截面模型超参**（`xgboost_cross_section.py`）：已通过正确的 Val/Test 划分完成调优，无需变动。

2. **策略超参的现实处理**：
   - `mktcap_pct_cut`：固化为 10%（搜索结果显示无效）
   - `stop_loss=8%`：维持（5%~10% 均稳健，8% 是合理中值）
   - `half_slots=3`：维持（敏感性低）
   - `max_slots=8`：维持（4槽在 Val 更好，但容量太小、集中度风险高）
   - `ma_death_days=5`, `min_hold_days=5`：维持（高度 regime 敏感，单年 val 不可信）

3. **下一步（若要改进）**：将策略 val 期扩展到 2022-2023，同时将 test 期移至 2024-2026，并采用 Robustness Criterion（最差子期最大化）作为优化目标。

---

## 十、输出文件

| 文件 | 说明 |
|------|------|
| `output/strategy_hp_search/hp_search_results.csv` | 所有 ~30 次 val 回测结果明细 |
| `output/strategy_hp_search/sensitivity_results.csv` | 敏感性测试：每个参数各值的 Calmar 变化 |
| `output/strategy_hp_search/best_strategy_params.json` | 最优参数 + val/test/default 三组绩效 |
| `output/strategy_hp_search/strategy_hp_equity.png` | Val 和 Test 的净值曲线（vs CSI300） |
| `output/strategy_hp_search/strategy_hp_sensitivity.png` | 各参数敏感性条形图 |
| `output/csv/xgb_cs_pred_val.csv` | Val 期截面预测（train-only 模型，供策略搜索使用） |
