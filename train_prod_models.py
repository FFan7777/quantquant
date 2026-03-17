#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产模型训练脚本
================
使用 2018-2025-12-31 全量数据训练所有模型，供日常实盘推理使用。
与 eval 模型（Train 2018-2021 / Val 2022-2024 / Test 2025+）完全独立。

超参来源：
  - CS H10/H5: 读取 eval 模型的 best_params 和 n_estimators（从已保存 JSON/模型文件）
  - 指数择时:  同 eval 模型配置（n_estimators=150, max_depth=2），2025年作为内部 ES val

输出文件：
  output/models/xgb_h10_prod.json
  output/models/lgb_h10_prod.txt
  output/models/features_h10_prod.json
  output/models/xgb_h5_prod.json
  output/models/features_h5_prod.json
  output/csv/index_timing_predictions_prod.csv

用法：
  python train_prod_models.py          # 训练所有生产模型（约 60 分钟）
  python train_prod_models.py --cs     # 仅训练 CS 模型（H10 + H5）
  python train_prod_models.py --timing # 仅训练指数择时模型
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
ROOT   = Path(__file__).parent


def run(cmd: list, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  [ERROR] {label} 失败（returncode={result.returncode}）")
        sys.exit(result.returncode)
    print(f"\n  [OK] {label} 完成，耗时 {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(description='训练生产模型（2018-2025 全量数据）')
    parser.add_argument('--cs',     action='store_true', help='仅训练 CS 模型（H10 + H5）')
    parser.add_argument('--timing', action='store_true', help='仅训练指数择时模型')
    args = parser.parse_args()

    run_cs     = args.cs or (not args.cs and not args.timing)
    run_timing = args.timing or (not args.cs and not args.timing)

    t_start = time.time()
    print("\n" + "="*60)
    print("  生产模型训练流水线")
    print("  训练数据: 2018-01 ~ 2025-12-31（全量）")
    print("="*60)

    if run_cs:
        run([PYTHON, "xgboost_cross_section.py", "--prod"],
            "H10 CS 模型（XGB + LGB，预计 ~30 分钟）")

        run([PYTHON, "xgboost_cross_section_h5.py", "--prod"],
            "H5 CS 模型（XGB，预计 ~10 分钟）")

    if run_timing:
        run([PYTHON, "index_timing_model.py",
             "--label_type", "ma60_state", "--no_wfo", "--prod"],
            "指数择时模型（预计 ~2 分钟）")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  所有生产模型训练完成，总耗时 {total:.0f}s")
    print(f"  infer_today.py 将自动优先使用 _prod 模型文件")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
