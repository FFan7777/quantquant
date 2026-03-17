# 开发实践与方法论

本文档记录量化研究过程中形成的核心开发规范、方法论和工程经验，供后续开发参考。

---

## 一、数据划分规范（防 Leakage 铁律）

### Train / Val / Test 三段分离

```
XGBoost 模型训练:  2018-01 ~ 2021-12   (LONG_VAL=True 时)
策略 Val（HP调优）: 2022-02 ~ 2024-12   (含 20日 embargo 隔离)
策略 Test（OOS）:  2025-02 ~ 2026-xx   ← 唯一真实评测，不可用于调参
```

**严禁**：
- 通过观察完整回测结果（包括 Test 期）来手工调整策略参数 → 这是 test-set leakage
- 在 Val 期后再次运行 HP 搜索（等价于将 Val 变成 Test）
- `LONG_VAL=False` 时 Val 仅 2024 单年 → 导致 Regime Overfitting（见第三节）

### 模型训练的 PIT（Point-In-Time）原则

- 财务数据必须使用 `f_ann_date`（第一披露日）而非 `ann_date`（数据库入库日）
- `f_ann_date` 通过 `np.searchsorted` 实现，确保每个 `trade_date` 只能看到 ≤ 该日已公布数据
- Val 期的 CS 预测必须用 **train-only 模型**（而非 train+val 模型）生成，保存至 `output/csv/xgb_cs_pred_val.csv`（H5 对应 `xgb_cs_pred_val_h5.csv`）

### 两阶段训练（H10/H5 共用模式）

```
Stage 1: 仅用 train(2018-2021) 训练，val(2022-2024) 做 Early Stopping
         → best_iteration 确定最优轮数
         → 在 val 上做推理 → 保存至 xgb_cs_pred_val*.csv（供 HP Search）
Stage 2: train+val 合并，固定 best_iteration 轮重训（无 Early Stopping）
         → model_final → 保存至 xgb_*.json（供 test 期推理/实盘）
```

**严禁**：用 model_final 生成 val 期预测 → model_final 见过 val 数据，会导致 HP Search leakage。

### H5 模型历史 Bug（2026-03-17 修复）

- **Bug**：`xgboost_cross_section_h5.py` 原 `TRAIN_END = "20250630"`，训练集包含整个 val+test 期 → H5 预测在回测中严重穿越
- **修复**：`TRAIN_END = "20211231"`，与 H10（LONG_VAL=True）对齐；新增两阶段训练
- **影响**：修复后 Test(2025+) Calmar 从 4.50 降至 3.54（旧数字含 leakage，不可信）

### Embargo（隔离期）

- Train/Val、Val/Test 边界各有 **20 个交易日** 隔离期
- 防止标签（未来收益）与特征（当日数据）通过时序相关性泄漏

---

## 二、超参搜索方法论

### 坐标下降搜索顺序

```
Step A: (max_slots, half_slots)         9种组合
Step B: stop_loss                       7种候选值
Step C: (bear_ma_death, bear_min_hold)  8种组合  ← 对熊市年份影响最大
Step D: (bull_ma_death, bull_min_hold)  8种组合
Step E: (neutral_ma_death, neutral_min_hold) 7种组合
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
合计 ~39 次回测，耗时约 60 秒
```

**关键 bug 防范**：搜索变量时必须用 dict override，而非重建 StrategyParams 对象：
```python
# ✓ 正确：dict 合并，只覆盖当前搜索维度
candidate = {**asdict(base_params), **{"stop_loss": s}}
# ✗ 错误：StrategyParams(stop_loss=s) 会将其他所有字段重置为默认值
```

### Minimax 目标函数（取代最大化整体 Calmar）

```python
def minimax_score(equity_df) -> float:
    """
    最大化 Val 期内最差年份的 Calmar 比率
    - 某年亏损 > 10%  → 重惩罚（-10）
    - 某年亏损 0~10%  → 轻惩罚（-2）
    - 全年正收益      → min(年度 Calmar)
    """
```

**原因**：最大化整体 Calmar 会使优化器"牺牲"某一年换取其他年的高收益，导致对某种市场环境完全失效。Minimax 强制找到在所有 regime 下都"说得过去"的参数。

### 多年 Val 防 Regime Overfitting

- **错误做法**：Val = 单年 2024（弱市/震荡）→ 优化出"适合熊市的参数"→ 2025 牛市大幅落后
- **正确做法**：Val = 2022-2024（熊市 + 弱势 + 震荡），覆盖多种 market regime

### Regime-Switching 退出参数

不同市场状态使用不同的 MA死叉/最短持有期：

| 市场状态 | slots | MA死叉天数 | 最短持有期 | 含义 |
|----------|-------|-----------|-----------|------|
| 牛市 | 20 | 5天 | 5天 | 容忍回调，让利润奔跑 |
| 中性/半仓 | 10 | 3天 | 5天 | 均衡，等待二次确认 |
| 熊市/空仓 | 0 | 2天 | 2天 | 立即响应，快速止损 |

**关键发现**：`bear_ma_death=2` 是将 2022 年从 -11% 变为 +1.7% 的核心参数。但注意 `bear_ma_death=3,4` 存在非单调不稳定区间（灵敏度范围 2.24），选 2 最为保守稳健。

---

## 三、已废弃研究方向（禁止重复）

### 未奏效的策略参数

见 README.md「失败实验存档」，不再重复。

### 未奏效的模型方向

| 方向 | 结论 |
|------|------|
| Per-stock 择时（xgboost_market_timing.py） | 将个股时序信号汇总为市场择时（v1 架构）会混淆空间与时序信息，导致信号衰减。已废弃，改用纯指数择时 |
| WFO（Walk-Forward Optimization） | ICIR 仅 +0.005，计算耗时 10x。收益不值得，保持固定训练集切分 |
| H20 三模型 Ensemble | H20 标签与 5 日调仓时域不匹配，引入噪声。保持 H10+H5 两模型融合 |

---

## 四、工程避坑记录

### pandas 3.0 / XGBoost 3.x 兼容性

```python
# stack(dropna=False) 已移除 → 改用 stack()
df.stack()                          # 而非 df.stack(dropna=False)

# merge_asof 要求数值型 join key → 改用 np.searchsorted 实现 PIT join

# groupby().apply() 会移走 group key 到 index → 改用显式 for 循环

# pd.get_dummies 默认返回 bool → 加 dtype=float
pd.get_dummies(df["industry"], dtype=float)
```

### CS 模型日期常量检查清单

每次新建或修改 CS 模型脚本时，确认：

```python
# ✓ TRAIN_END  ≤ 2021-12-31（LONG_VAL=True 模式）
# ✓ VAL_START  = TRAIN_END + 20 交易日 embargo
# ✓ TEST_START = VAL_END + 20 交易日 embargo
# ✓ Val 预测使用 model_train_only（非 model_final）
# ✗ 绝不把 TEST_START 设成未来或当前日期（如 20250630）→ 严重 leakage
```

### XGBoost 非确定性

- `n_jobs=-1` 多线程导致浮点计算顺序不同，同一代码每次运行年化可差 **±2%（Sharpe ±0.1）**
- **缓解方案**：LGB Ensemble（0.5×XGB rank + 0.5×LGB rank 平均）可部分平滑噪声
- **完全复现**：设 `n_jobs=1`（慢约 5x）
- 记录模型结果时需说明随机性范围，不要用单次结果做决策

### DuckDB 查询规范

```python
# ✓ 始终用 trade_date 字符串格式 YYYYMMDD，不要混用日期格式
# ✓ 批量 INSERT 用 "INSERT OR REPLACE INTO t SELECT * FROM df"
# ✓ 只读连接：duckdb.connect(db_path, read_only=True)
# ✗ 避免长时间保持写连接（会阻塞其他读操作）
```

### macOS 环境依赖

```bash
brew install libomp   # XGBoost macOS 需要（否则报 dylib 错误）
```

---

## 五、模型更新流程

### 何时重训

- 数据新增 **> 3 个月**（约每季度一次）
- 重大市场 regime 转换后（如从持续熊市进入牛市）

### 更新顺序（必须严格遵守）

```
1. python main.py update                  # 确保数据最新
2. python xgboost_cross_section.py        # H10 CS 模型（输出模型 + 预测 CSV）
3. python xgboost_cross_section_h5.py # H5 CS 模型
4. python index_timing_model.py --label_type ma60_state --no_wfo  # 指数择时
5. python strategy_hp_search.py           # 策略超参重新搜索（Val 2022-2024，Minimax）
6. 将 HP Search 输出的最优参数更新到 index_ma_combined_strategy.py 和 infer_today.py
7. python index_ma_combined_strategy.py   # 验证 Test 期绩效
```

**注意**：步骤 5 产生的最优参数必须同时更新到 `index_ma_combined_strategy.py`（回测）和 `infer_today.py`（实盘），确保参数一致。

### LONG_VAL 标志

- `LONG_VAL=True`（推荐）：Train=2018-2021，Val=2022-2024，防止 Regime Overfitting
- `LONG_VAL=False`：Train=2018-2023，Val=2024（不推荐，已证明导致 Regime Overfitting）

---

## 六、数据收集注意事项

### Tushare API 限速

- 单次请求间隔 ≥ 0.2s（低积分用 0.5s）
- 并发线程数：积分 < 120 → 3，120-2000 → 5，2000+ → 10
- `_retry_request` 方法自动处理超限重试（最多 3 次，指数退避）

### 财务数据 f_ann_date

- `income_statement.f_ann_date`：第一披露日，真实可知时点，用于 PIT join ✅
- `fina_indicator.ann_date`：大部分是 2023+ 的数据库入库日期，**不是披露日** ❌
- F-Score 必须从 income_statement + balance_sheet + cash_flow 三表自行计算

### stk_holdernumber（股东户数）

- 按季度末日期批量查询（每次约 5000 行）
- 使用 `ann_date`（公告日）作为 PIT 时点，在 end_date（季度末）之后约 20-60 天
- 命令：`python main.py collect-holder`

---

## 七、代码时间穿越检查清单

每次添加新特征时，逐项确认：

- [ ] 特征计算只使用 ≤ trade_date 的数据
- [ ] 财务数据使用 f_ann_date（非 ann_date / end_date）
- [ ] 标签（未来收益）使用 shift(-HORIZON)，仅用于训练标签，不作为特征
- [ ] 截面中性化（Ridge 回归）在每个截面内独立计算，不跨截面使用
- [ ] 分析师特征 `report_date <= trade_date`
- [ ] 停牌/涨停过滤使用 T 日当日数据（vol > 0 / pct_chg）
- [ ] 训练/测试切分有 20 日隔离期
