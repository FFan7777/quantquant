#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集股东户数数据 (stk_holdernumber) 并存入 DuckDB
按季度末日期批量查询，单次获取全市场数据（~5000行/次，~40次查询共需几分钟）
运行: python collect_holder_data.py
"""
import sys, time
from pathlib import Path
import duckdb
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
from data_collect.tushare_api import TushareAPI

DB_PATH    = str(ROOT / "data/quant.duckdb")
START_YEAR = 2016
END_YEAR   = 2026

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS stk_holdernumber (
    ts_code    VARCHAR NOT NULL,
    ann_date   VARCHAR,
    end_date   VARCHAR NOT NULL,
    holder_num BIGINT,
    PRIMARY KEY (ts_code, end_date)
)
"""

def quarter_end_dates(start_year: int, end_year: int) -> list[str]:
    dates = []
    for y in range(start_year, end_year + 1):
        for mmdd in ["0331", "0630", "0930", "1231"]:
            d = f"{y}{mmdd}"
            if d <= "20260311":
                dates.append(d)
    return dates


def main():
    api = TushareAPI()

    with duckdb.connect(DB_PATH) as conn:
        conn.execute(CREATE_SQL)
        # 查询已有季度
        existing = set(conn.execute(
            "SELECT DISTINCT end_date FROM stk_holdernumber"
        ).fetchdf()["end_date"].tolist())

    dates = quarter_end_dates(START_YEAR, END_YEAR)
    todo  = [d for d in dates if d not in existing]
    print(f"待收集季度: {len(todo)} 个（已有: {len(existing)} 个）")
    print(f"  日期范围: {dates[0]} ~ {dates[-1]}")

    total_rows = 0
    with duckdb.connect(DB_PATH) as conn:
        for end_date in tqdm(todo, desc="收集股东户数（按季度）"):
            try:
                df = api._retry_request(
                    api.pro.stk_holdernumber,
                    end_date=end_date,
                    fields="ts_code,ann_date,end_date,holder_num"
                )
                if df is None or df.empty:
                    continue
                df["holder_num"] = pd.to_numeric(df["holder_num"], errors="coerce")
                df = df.dropna(subset=["ts_code", "end_date", "holder_num"])
                df["holder_num"] = df["holder_num"].astype(int)
                if not df.empty:
                    conn.execute("INSERT OR REPLACE INTO stk_holdernumber SELECT * FROM df")
                    total_rows += len(df)
            except Exception as e:
                print(f"  [{end_date}] 失败: {e}")

    print(f"\n完成！共入库 {total_rows:,} 条记录")
    with duckdb.connect(DB_PATH, read_only=True) as conn:
        r = conn.execute("""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT ts_code) AS stocks,
                   MIN(end_date) AS min_date,
                   MAX(end_date) AS max_date
            FROM stk_holdernumber
        """).fetchone()
    print(f"  stk_holdernumber: {r[0]:,} 行, {r[1]} 只, {r[2]}~{r[3]}")


if __name__ == "__main__":
    main()
