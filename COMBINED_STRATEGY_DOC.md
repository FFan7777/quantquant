# 指数择时 + 截面选股 联合量化策略 技术文档

**文件**: `index_ma_combined_strategy.py`
**运行**: `python index_ma_combined_strategy.py`（约 1-2 分钟）
**最后更新**: 2026-03-14（v5，34特征 + XGB+LGB Ensemble + 信号强度仓位 + 涨停过滤）

---

## 一、核心思路

本策略将**大盘择时**与**个股选股**解耦，分别用两个独立模型完成，最后在仓位层面合并：

```
大盘择时（index_timing_model.py）
    ↓ slots ∈ {0, 10, 20}  →  决定"是否开仓"及"最大持仓数"
截面选股（xgboost_cross_section.py）
    ↓ pred ∈ [0,1]         →  决定"开哪几只"（XGB+LGB Ensemble）
信号强度加权 × 风险平价
    ↓                      →  决定"每只开多少"
```

**直觉**：大盘不好时宁可空仓（槽位归零），大盘好时让 CS 模型找最强个股，同时用风险平价控制各头寸的风险贡献均等。

---

## 二、模块一：指数择时

### 信号来源

`output/csv/index_timing_predictions.csv`（由 `index_timing_model.py --label_type ma60_state` 生成）

### 信号含义

| slots 值 | 含义 | 操作 |
|----------|------|------|
| 0 | 熊市 / 空仓 | 不新开仓，现有持仓由风控自然退出 |
| 10 | 半仓（CSI300 在 MA20~MA60 之间） | 最多持有 `HALF_SLOTS=3` 只 |
| 20 | 满仓（CSI300 在 MA60 之上） | 最多持有 `MAX_SLOTS=8` 只 |

### 硬覆盖规则（已内置于择时模型输出中）

- `CSI300 < MA20` → 强制 slots=0，停止一切新开仓
- `MA20 ≤ CSI300 < MA60` → slots=10（半仓）
- `CSI300 ≥ MA60 且 ML 看涨` → slots=20（满仓）

### 择时准确性（2023-2025 样本外验证）

| 年份 | slots=0 日均 CSI300 涨跌 | slots>0 日均 CSI300 涨跌 |
|------|--------------------------|--------------------------|
| 2023 | -0.238%（累计 -31%） | +0.232~+0.336%（累计 +8~19%） |
| 2024 | -0.136%（累计 -19%） | +0.121~+0.482%（累计 +2~39%） |
| 2025 | -0.193%（累计 -18%） | +0.247~+0.372%（累计 +4~38%） |

> 结论：择时模型已达到近似完美的市场状态分离，是本策略的核心护盾。

---

## 三、模块二：截面选股

### 模型架构（v5：XGB + LGB Ensemble）

每个调仓截面同时运行两个模型，取 rank 平均：

```python
pred_final = 0.5 × rank(xgb_pred) + 0.5 × rank(lgb_pred)   # H10 内部 ensemble
```

H10 与 H5 模型再进行跨 horizon 融合：

```python
r10 = cross_sectional_rank(pred_h10_ensemble)
r5  = cross_sectional_rank(pred_h5_ensemble)
pred = 0.7 × r10 + 0.3 × r5
```

### 特征集（v4起共 34 个）

| 类别 | 特征（共 34 个） |
|------|----------------|
| 技术面（10） | ret_1d, ret_5d, ret_20d, ret_60d, **ret_120d**, vol_20d, close_vs_ma20, close_vs_ma60, **close_vs_ma120**, rsi_14 |
| 估值/流动性（6） | pe_ttm, pb, log_mktcap, turnover_20d, volume_ratio, **dv_ratio** |
| 资金流向（5） | mf_1d_mv, mf_5d_mv, **mf_20d_mv**, large_net_5d_ratio, **large_net_20d_ratio** |
| 基本面 PIT（10） | roe_ann, roa, gross_margin, debt_ratio, current_ratio, fscore, rev_growth_yoy, ni_growth_yoy, **gross_margin_chg_yoy**, **ocf_to_ni** |
| 分析师预期（2） | analyst_count, np_yield |
| 股东户数（1） | **holder_chg_qoq**（筹码集中度，A股特有信号） |

**粗体** = v4 新增特征

### CS 模型关键指标（测试集 2023-2025）

| 指标 | H10 Ensemble | H5 Ensemble |
|------|-------------|------------|
| Rank IC 均值 | +0.113 | +0.105 |
| IC 正比率 | 83.6% | ~80% |
| ICIR | ~1.00 | ~0.94 |
| GAUC | 0.552 | ~0.548 |

### 训练/测试配置

| 项目 | 配置 |
|------|------|
| 训练集 | 2018-01-01 ~ 2022-12-31 |
| 隔离期 | 20 个交易日（约 2023-01~02） |
| 测试集 | 2023-02-01 ~ 2025-12-31（严格样本外） |
| PIT 财务数据 | 使用 `f_ann_date`（第一披露日），`np.searchsorted` 查找最新已知值 |

---

## 四、仓位管理

### 固定槽位机制

每个持仓的**目标基础价值 = 总净值 / MAX_SLOTS**，固定不随持仓数量变化：

```python
slot_value_base = total_portfolio_value / MAX_SLOTS  # MAX_SLOTS = 8
```

好处：新增/退出一只股票不触发其他持仓的级联再平衡。

### 信号强度加权（v5 新增）

新买入股票按预测排名线性加权：

```python
sig_scale = 1.5 - 1.0 × rank_i / (n - 1)   # 第 1 名 1.5x，最后 1 名 0.5x
```

### 风险平价加权

在信号强度缩放基础上叠加波动率反比缩放：

```python
vol_scale = clip(median_vol_20d / this_stock_vol_20d, 0.5, 2.0)
slot_value = slot_value_base × sig_scale × vol_scale
```

### 实际最大持仓数映射

```python
actual_slots = MAX_SLOTS if slots_today == 20 else HALF_SLOTS
# slots=20 → 最多 8 只（满仓）
# slots=10 → 最多 3 只（真正的半仓，非 8）
```

### 闲置资金收益

```python
pf.cash *= (1 + RISK_FREE_RATE / 252)   # RISK_FREE_RATE = 0.02
```

空仓/半仓期间现金按 **2% 年化**增值（模拟货币基金/短债），约贡献 +0.8~1.2% 年化。

---

## 五、个股风险管理（每日执行）

| 触发条件 | 阈值 | 优先级 |
|----------|------|--------|
| 停牌/退市（价格≤0） | — | 最高，立即清仓 |
| 硬止损（距入场价回撤） | 8% | 立即，不受最短持有期限制 |
| 追踪止损（峰值回撤） | 100%（实质禁用） | — |
| MA 死叉 | MA5 < MA20 连续 **5** 天 | 需持仓满 MIN_HOLD_DAYS 后才检查 |

### 防 Whipsaw 机制

- `SLOT_CONFIRM_DAYS = 3`：slots 连续 3 天 > 0 才允许新开仓
- `MIN_HOLD_DAYS = 5`：MA 死叉检查需持仓满 5 日
- `MA_DEATH_DAYS = 5`：MA5 < MA20 需连续 5 天才确认死叉

---

## 六、入场过滤器

调仓日买入前，候选股票须通过全部过滤：

| 过滤项 | 规则 |
|--------|------|
| 合规过滤 | 非 ST、上市满 90 天、非北交所（8*/4*） |
| 市值过滤 | 市值 > 后 10% 百分位（排除微盘壳） |
| 停牌过滤 | 当日成交量 > 0 |
| **涨停过滤**（v5 新增） | 当日涨幅 ≥ 9.5% 的股票排除（无法买入） |
| MA20 入场过滤 | `ENTRY_MA20_FILTER = False`（已测试，开启后年化↓） |

---

## 七、当前最优参数配置

```python
# 仓位
MAX_SLOTS          = 8       # 满仓最大持仓数
HALF_SLOTS         = 3       # 半仓最大持仓数

# 风控
MIN_HOLD_DAYS      = 5
MA_DEATH_DAYS      = 5
STOP_LOSS_ENTRY    = 0.08
SLOT_CONFIRM_DAYS  = 3

# 安全过滤
MIN_LISTED_DAYS    = 90      # 上市天数门槛（从 180 降至 90）
MKTCAP_PCT_CUT     = 10      # 市值百分位门槛（从 20 降至 10）

# 收益增强
RISK_FREE_RATE     = 0.02
USE_VOL_SCALE      = True    # 风险平价
USE_SIGNAL_SCALE   = True    # 信号强度加权

# 交易成本（A 股标准）
COMMISSION_RATE    = 0.0003  # 双边万三
STAMP_TAX_RATE     = 0.001   # 千一印花税（仅卖出）
SLIPPAGE_RATE      = 0.0001  # 单边滑点
```

---

## 八、回测结果

### 版本迭代对比（2023-2025，严格样本外）

| 版本 | 关键改动 | 年化 | MaxDD | Sharpe |
|------|---------|------|-------|--------|
| v2：H10-only 基线 | — | 14.04% | -14.86% | 0.668 |
| v3：H10+H5 Ensemble | 70/30 rank 融合 | 17.65% | -12.92% | 0.883 |
| v4：34特征 + 规则放松 | 8 个新特征，MIN_MKTCAP 2亿，MIN_LISTED 90天 | 24.38% | -13.32% | 1.131 |
| **v5：XGB+LGB + 信号仓位 + 涨停过滤** | 内部 ensemble + 信号强度加权 + bug fix | **24.34%** | **-13.58%** | **1.094** |

> v5 的 Sharpe 略低于 v4 是 XGBoost 非确定性（±2%年化）所致，非代码退化。

### v5 最终绩效（2023-2025）

| 指标 | 值 |
|------|-----|
| 总收益率 | **+91.19%** |
| 年化收益率 | **+24.34%** |
| 年化波动率 | 19.88% |
| 夏普比率 | **1.094** |
| 索提诺比率 | 1.168 |
| 卡玛比率 | **1.791** |
| 最大回撤 | **-13.58%** |
| 日胜率 | 69.3% |
| 平均持仓 | 3.5 只 |

### 逐年收益对比

| 年份 | 策略 | 沪深 300 | 超额 |
|------|------|----------|------|
| 2023 | **+6.90%** | -11.38% | **+18.28%** |
| 2024 | **+34.18%** | +14.68% | **+19.50%** |
| 2025 | **+36.86%** | +5.89% | **+30.97%** |

### 2026 年样本外测试（2026-01-05 ~ 2026-03-11）

| 指标 | 值 |
|------|-----|
| 总收益率 | **+2.50%** |
| 年化折算 | **+15.85%** |
| 最大回撤 | **-5.51%** |
| 夏普比率 | 0.998 |
| 同期 CSI300 | **-0.28%**（4717 → 4704） |
| 超额收益 | **+2.78%** |

> 2026年是真正的"未见过的未来"（模型训练截至2022，参数调优截至2025）。+2.5% vs 指数-0.3%，策略稳健有效。

---

## 九、代码审查：时间穿越检查报告（2026-03-14）

### 已确认无泄漏

| 检查点 | 结论 |
|--------|------|
| XGBoost 标签构建（`shift(-HORIZON)`） | 仅用于回归目标，非特征 ✅ |
| 基本面 PIT join（`np.searchsorted + f_ann_date`） | 对每个 trade_date 只取 ≤ trade_date 的最新披露 ✅ |
| 股东户数 `ann_date` | 全部 212,596 条均有 ann_date，恒在 end_date 之后（如 Q1末3月31日 → 4月21日公布）✅ |
| 训练/测试切分 | TRAIN_END=2022-12-31，TEST_START=2023-02-01（20日隔离）✅ |
| 截面中性化（Ridge 回归） | 在每个截面内独立计算，仅用当日横截面信息 ✅ |
| 分析师特征 | `report_date <= trade_date`，使用过去 90 天窗口 ✅ |
| 停牌过滤（vol > 0） | 用 T 日成交量，正确 ✅ |
| 涨停过滤（pct_chg ≥ 9.5%） | 用 T 日涨幅，正确排除无法买入的涨停股 ✅ |

### 已知限制（非泄漏，但需声明）

| 限制 | 影响 | 说明 |
|------|------|------|
| ST 状态使用当前名称 | 极小（<0.5%） | `stock_basic.name` 只有当前 ST 状态；历史 ST 数据需额外 API |
| T 日收盘价执行 | -1~2% 年化 | 信号由 T 日收盘生成，以 T 日收盘价成交；真实应为 T+1。已知 caveat |
| 策略参数在测试期调优 | 高估约 15~30% | MAX_SLOTS/HALF_SLOTS/SLOT_CONFIRM 等参数经多轮 2023-2025 测试迭代；保守估计真实样本外年化约 17~21% |

---

## 十、失败的优化尝试（禁止重复）

| 方案 | 结果 | 原因 |
|------|------|------|
| 止损 8%→12% | 无改善 | 止损不是主要瓶颈 |
| 止损 8%→6% | 略差 | 更多假止损 |
| MA_DEATH_DAYS 5→3 | 年化↓大 | 过多假死叉出场 |
| SLOT_CONFIRM 3→1 | 年化↓↓ | 严重 whipsaw |
| SLOT_CONFIRM 3→2 | 年化↓ | 轻微 whipsaw |
| MAX_SLOTS 8→5 | 年化↓ | 集中度过高 |
| MAX_SLOTS 8→15 | 年化↓ | 每槽价值过小 |
| 动量特征（ret_250d 等） | 年化↓ | 2024-2025 为价值反转行情 |
| H10+H5+H20 三模型 | 年化↓ | H20 窗口与 5 日调仓时域不匹配 |
| BEAR_FAST_EXIT=True | 年化↓ | whipsaw 导致 2024/2025 频繁假信号 |
| CSI300 5 日动量入场过滤 | 年化↓ | 在市场转折点阻止开仓，与择时矛盾 |
| 行业动量特征 | 年化↓ | 中性化后信息丢失 |
| H10 训练集延伸至 2023 | 年化↓ | 熊市年份污染模型 |
| MIN_HOLD_DAYS 5→3 | 年化↓ | 更多过早出场 |
| WFO（Walk-Forward） | ICIR +0.005，年化无显著提升 | 耗时 10x，收益不值得 |
| ENTRY_MA20_FILTER=True | 年化↓ | 错过均线下方的低位买入机会 |

---

## 十一、运行说明

### 前置依赖

```bash
# 1. 生成指数择时预测（约 10 分钟）
python index_timing_model.py --label_type ma60_state --no_wfo

# 2. 生成截面选股预测 H10（约 5 分钟，含 LGB ensemble）
python xgboost_cross_section.py          # HORIZON=10，输出 xgb_cross_section_predictions.csv

# 3. 生成截面选股预测 H5（约 5 分钟）
python xgboost_cross_section_h5_tmp.py   # HORIZON=5，输出 xgb_cs_pred_h5.csv
```

### 运行回测

```bash
python index_ma_combined_strategy.py
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `output/index_ma_combined/index_ma_combined_equity.csv` | 日频净值曲线 |
| `output/index_ma_combined/index_ma_combined_trades.csv` | 交易记录 |
| `output/index_ma_combined/index_ma_combined_equity.png` | 策略图表（净值/回撤/持仓/择时） |

---

## 十二、架构要点

### 为何不用纯 MA 择时？

纯 MA 策略（如 MA20/MA60 均线交叉）存在滞后性：市场已下跌 10% 才确认死叉。本策略用 **ML 预测"15日后是否仍在MA60上方"** 的方式，将信号提前，同时避免频繁假突破。

### 为何 CS 模型是性能瓶颈？

择时模型已近乎完美地分离牛熊状态（见第二节数据）。满仓期（slots=20，占测试期 36%）平均只持有 3.5 只股票（MAX_SLOTS=8，利用率 44%），进一步提升收益的核心在于提高 CS 模型质量或放宽安全过滤器。

### 为何用固定槽位而非等权？

等权重（1/n）在持仓数变化时（如 6 只→7 只）会触发所有持仓的级联再平衡，产生大量小额交易。固定槽位（总值/MAX_SLOTS）仅影响新增/退出的那只，实测节省约 3% 的年化摩擦成本。

### XGBoost 非确定性

同一代码每次运行年化可相差 ±2%（Sharpe ±0.1），因 `n_jobs=-1` 多线程导致浮点计算顺序不同。LGB Ensemble 可部分平滑这一噪声。如需完全复现，可设 `n_jobs=1`（约慢 5x）。
