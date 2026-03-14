#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强型基本面策略 v3（时序版，无未来信息）

数据来源：income_statement + balance_sheet + cash_flow（f_ann_date 为第一披露日）
         + moneyflow（资金流向）+ daily_basic（市值/PE）

买入条件（全部满足）:
  1. 市值 30亿–500亿
  2. F-Score >= 6（Piotroski 9维，同比 YoY，从原始报表计算）
  3. ROE > 10%
  4. MA5 > MA20（价格趋势向上）
  5. RSI(14) < 60（动量未过热）
  6. 量比 > 1.5（成交量放大）
  7. PE 横截面百分位 < 80%

持有条件（任一违反则清仓，不检查 RSI/量比/PE）:
  - F-Score >= 6  AND  ROE > 10%
  - MA5 连续 3 日 < MA20（死叉确认，减少假死叉）
  - 持有期最高点回撤 < 7%（立即触发）
  - 价格 > 成本价 × 90%（立即触发）

仓位控制（三档，基于上证指数）:
  - SSE >= MA60 → 满仓（top_n 槽位）
  - MA20 <= SSE < MA60 → 半仓（top_n/2 槽位）
  - SSE < MA20 → 停止新增（现有持仓由持有条件自然退出）

评分系统（top_n 选股，固定槽位权重 1/top_n）:
  - 基本面 40%: F-Score/9×33%  +  min(ROE/30,1)×45%  +  (1-PE_pct/100)×22%
  - 技术面 30%: RSI信号×35%  +  MA强度×30%  +  价格动量×35%
  - 资金面 30%: 量比×40%  +  净资金流入排名×60%
"""

import numpy as np
import pandas as pd
from typing import cast, Dict, Optional, Set, Tuple, Union

from .base_strategy import BaseStrategy


class EnhancedFundamentalStrategy(BaseStrategy):
    """增强型基本面策略 v3"""

    def __init__(
        self,
        top_n: int = 20,
        min_fscore: int = 6,
        max_pe_percentile: float = 80.0,
        min_roe: float = 10.0,
        ma_short: int = 5,
        ma_long: int = 20,
        rsi_period: int = 14,
        rsi_threshold: float = 60.0,
        vol_ratio_period: int = 5,
        vol_ratio_threshold: float = 1.5,
        stop_loss_pct: float = 0.07,
        cost_stop_loss_pct: float = 0.10,
        min_mktcap: float = 30.0,          # 亿元
        max_mktcap: float = 500.0,         # 亿元
        # 评分权重（基本面/技术面/资金面各子项）
        w_fscore: float = 0.33,
        w_roe: float = 0.45,
        w_pe: float = 0.22,
        w_rsi: float = 0.35,
        w_ma: float = 0.30,
        w_mom: float = 0.35,
        w_volratio: float = 0.40,
        w_mf: float = 0.60,
        name: str = "EnhancedFundamental_v3",
    ):
        super().__init__(name)
        self.top_n = top_n
        self.min_fscore = min_fscore
        self.max_pe_percentile = max_pe_percentile
        self.min_roe = min_roe
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.vol_ratio_period = vol_ratio_period
        self.vol_ratio_threshold = vol_ratio_threshold
        self.stop_loss_pct = stop_loss_pct
        self.cost_stop_loss_pct = cost_stop_loss_pct
        self.min_mktcap = min_mktcap
        self.max_mktcap = max_mktcap
        self._w = dict(
            fscore=w_fscore, roe=w_roe, pe=w_pe,
            rsi=w_rsi, ma=w_ma, mom=w_mom,
            volratio=w_volratio, mf=w_mf,
        )

        # 止损跟踪
        self.holding_high: Dict[str, float] = {}
        self.entry_prices: Dict[str, float] = {}
        self.entry_dates: Dict[str, str] = {}   # ts_code → 建仓日期
        self.ma_cross_count: Dict[str, int] = {}  # ts_code → 连续死叉天数

        # 技术指标矩阵缓存
        self.price_matrix: Optional[pd.DataFrame] = None
        self._ma_short_matrix: Optional[pd.DataFrame] = None
        self._ma_long_matrix: Optional[pd.DataFrame] = None
        self._ma_mid_matrix: Optional[pd.DataFrame] = None   # MA60 用于动量
        self.vol_matrix: Optional[pd.DataFrame] = None
        self._rsi_matrix: Optional[pd.DataFrame] = None
        self._vol_ratio_matrix: Optional[pd.DataFrame] = None

        # 时序基本面（从 DB 加载）
        self._fundamental_sorted: Optional[Dict[str, Dict]] = None

        # 市值数据（ts_code → total_share 万股，用于历史缺失日期的回退）
        self._total_share: Optional[Dict[str, float]] = None

        # 历史市值矩阵（亿元，从 daily_basic.total_mv 加载）
        self._mktcap_matrix: Optional[pd.DataFrame] = None

        # 历史 PE TTM 矩阵（从 daily_basic.pe_ttm 加载）
        self._pe_matrix: Optional[pd.DataFrame] = None

        # 资金流向评分矩阵（date → ts_code → [0,1]）
        self._mf_score_matrix: Optional[pd.DataFrame] = None

        # 市场指数仓位控制
        self._index_series: Optional[pd.Series] = None
        self._index_ma20_series: Optional[pd.Series] = None
        self._index_ma60_series: Optional[pd.Series] = None
        self._index_max_pos: float = 0.5
        self._last_market_ratio: float = 1.0

        # 交易日列表
        self.trade_dates: list = []

    # ------------------------------------------------------------------
    # 价格矩阵预处理
    # ------------------------------------------------------------------

    def prepare_price_matrix(self, data: pd.DataFrame) -> None:
        """预处理价格数据为矩阵形式（仅首次执行）"""
        if self.price_matrix is not None:
            return
        self.price_matrix = data.pivot(
            index='trade_date', columns='ts_code', values='close'
        )
        print(f"✓ 价格矩阵缓存完成: {self.price_matrix.shape[0]} 天 x {self.price_matrix.shape[1]} 只股票")

    # ------------------------------------------------------------------
    # 一次性数据加载（入口）
    # ------------------------------------------------------------------

    def load_time_series_fundamentals(self, db_path: str) -> None:
        """
        从 DuckDB 加载全部所需数据：
        1. income_statement + balance_sheet + cash_flow → 时序基本面 + F-Score
        2. moneyflow → 资金流向评分矩阵
        3. daily_basic → 总股本（市值过滤）
        """
        import duckdb
        import time

        t0 = time.time()
        print("\n[基本面] 加载时序数据（无未来信息）...")

        conn = duckdb.connect(db_path, read_only=True)
        try:
            raw_df = self._build_from_raw_statements(conn)
            fscore_panel = self._build_fscore_panel(raw_df)

            # 构建二分查找结构
            self._fundamental_sorted = {}
            for ts_code, group in fscore_panel.groupby("ts_code"):
                g = group.sort_values("ann_date")
                self._fundamental_sorted[str(ts_code)] = {
                    "dates":         g["ann_date"].values,
                    "fscores":       g["fscore"].values.astype(float),
                    "roes":          g["roe"].values.astype(float),
                    "eps":           g["eps"].values.astype(float),
                    "period_months": g["period_months"].values.astype(int),
                }

            # 市值：优先使用 daily_basic 历史 total_mv 矩阵；无数据时用最新 total_share 近似
            self._load_total_share(conn)
            self._load_daily_basic_matrices(conn)

            # 资金流向评分矩阵
            self._load_moneyflow_matrix(conn)

        finally:
            conn.close()

        elapsed = time.time() - t0
        print(
            f"✓ 数据加载完成 ({elapsed:.1f}s): "
            f"{len(self._fundamental_sorted)} 只股票有财务数据"
        )

    # ------------------------------------------------------------------
    # 从三张原始报表构建财务指标
    # ------------------------------------------------------------------

    @staticmethod
    def _build_from_raw_statements(conn) -> pd.DataFrame:
        """
        合并 income_statement + balance_sheet + cash_flow，
        计算 ROA, ROE, 毛利率, 流动比率, 资产负债率, 资产周转率, OCF, EPS。
        使用 f_ann_date（第一披露日）作为信息可知日。
        """
        print("  [1/3] 加载原始财务报表...")

        df = conn.execute("""
            WITH is_t AS (
                SELECT ts_code, end_date,
                       MIN(COALESCE(f_ann_date, ann_date)) AS ann_date,
                       FIRST(revenue      ORDER BY ann_date DESC) AS revenue,
                       FIRST(oper_cost    ORDER BY ann_date DESC) AS oper_cost,
                       FIRST(n_income_attr_p ORDER BY ann_date DESC) AS n_income,
                       FIRST(basic_eps    ORDER BY ann_date DESC) AS eps
                FROM income_statement
                WHERE comp_type = '1'
                GROUP BY ts_code, end_date
            ),
            bs_t AS (
                SELECT ts_code, end_date,
                       FIRST(total_assets    ORDER BY ann_date DESC) AS total_assets,
                       FIRST(total_liab      ORDER BY ann_date DESC) AS total_liab,
                       FIRST(total_hldr_eqy_inc_min_int ORDER BY ann_date DESC) AS equity,
                       FIRST(total_cur_assets ORDER BY ann_date DESC) AS cur_assets,
                       FIRST(total_cur_liab   ORDER BY ann_date DESC) AS cur_liab
                FROM balance_sheet
                WHERE comp_type = '1'
                GROUP BY ts_code, end_date
            ),
            cf_t AS (
                SELECT ts_code, end_date,
                       FIRST(n_cashflow_act ORDER BY ann_date DESC) AS ocf
                FROM cash_flow
                WHERE comp_type = '1'
                GROUP BY ts_code, end_date
            )
            SELECT
                i.ts_code, i.end_date, i.ann_date,
                i.revenue, i.oper_cost, i.n_income, i.eps,
                b.total_assets, b.total_liab, b.equity,
                b.cur_assets, b.cur_liab,
                c.ocf
            FROM is_t i
            LEFT JOIN bs_t b USING (ts_code, end_date)
            LEFT JOIN cf_t c USING (ts_code, end_date)
            ORDER BY ts_code, end_date
        """).fetchdf()

        # 计算衍生指标（ROE 单位 %）
        ta = df["total_assets"].replace(0, np.nan)
        eq = df["equity"].replace(0, np.nan)
        rev = df["revenue"].replace(0, np.nan)

        df["roa"]               = df["n_income"] / ta
        df["roe"]               = df["n_income"] / eq * 100.0
        df["grossprofit_margin"]= (df["revenue"] - df["oper_cost"]) / rev * 100.0
        df["current_ratio"]     = df["cur_assets"] / df["cur_liab"].replace(0, np.nan)
        df["debt_to_assets"]    = df["total_liab"] / ta
        df["assets_turn"]       = df["revenue"] / ta

        print(f"  原始报表合并完成: {len(df):,} 条, {df['ts_code'].nunique()} 只股票")
        return df

    # ------------------------------------------------------------------
    # F-Score 面板（YoY 同比）
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fscore_panel(df: pd.DataFrame) -> pd.DataFrame:
        """
        构建 (ts_code, ann_date) → (fscore, roe, eps, period_months) 面板。
        F-Score 9 维（同比 YoY）:
          F1: ROA > 0          F2: OCF > 0
          F3: 净利润同比增长    F4: OCF/TA > ROA（现金质量）
          F5: 资产负债率↓      F6: 流动比率↑
          F7: 营收同比增长      F8: 毛利率↑
          F9: 资产周转率↑
        """
        print("  [2/3] 计算 F-Score 面板...")

        df = df.copy()
        df["ann_date"] = df["ann_date"].astype(str)
        df["end_date"] = df["end_date"].astype(str)
        df = df.sort_values(["ts_code", "end_date", "ann_date"])

        # 去重：每个 (ts_code, end_date) 保留最新修订
        df_dedup = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last").copy()

        # 计算同比（上一自然年同期）
        df_dedup["year"] = df_dedup["end_date"].str[:4].astype(int)
        df_dedup["mmdd"] = df_dedup["end_date"].str[4:]
        df_dedup["prev_end_date"] = (df_dedup["year"] - 1).astype(str) + df_dedup["mmdd"]

        prev_cols = {
            "roa":                "roa_p",
            "debt_to_assets":     "da_p",
            "current_ratio":      "cr_p",
            "grossprofit_margin": "gpm_p",
            "assets_turn":        "at_p",
            "n_income":           "ni_p",
            "revenue":            "rev_p",
        }
        df_prev = df_dedup[["ts_code", "end_date"] + list(prev_cols.keys())].rename(
            columns={"end_date": "prev_end_date", **prev_cols}
        )
        df_dedup = df_dedup.merge(df_prev, on=["ts_code", "prev_end_date"], how="left")

        # 9 个信号
        roa = df_dedup["roa"].fillna(0.0)
        ocf = df_dedup["ocf"].fillna(0.0)
        ta  = df_dedup["total_assets"].replace(0, np.nan)

        df_dedup["f1"] = (roa > 0).astype(int)
        df_dedup["f2"] = (ocf > 0).astype(int)
        df_dedup["f3"] = (
            df_dedup["n_income"].notna() & df_dedup["ni_p"].notna() &
            (df_dedup["n_income"] > df_dedup["ni_p"])
        ).astype(int)
        df_dedup["f4"] = (ocf / ta > roa).astype(int)
        df_dedup["f5"] = (
            df_dedup["debt_to_assets"].notna() & df_dedup["da_p"].notna() &
            (df_dedup["debt_to_assets"] < df_dedup["da_p"])
        ).astype(int)
        df_dedup["f6"] = (
            df_dedup["current_ratio"].notna() & df_dedup["cr_p"].notna() &
            (df_dedup["current_ratio"] > df_dedup["cr_p"])
        ).astype(int)
        df_dedup["f7"] = (
            df_dedup["revenue"].notna() & df_dedup["rev_p"].notna() &
            (df_dedup["revenue"] > df_dedup["rev_p"])
        ).astype(int)
        df_dedup["f8"] = (
            df_dedup["grossprofit_margin"].notna() & df_dedup["gpm_p"].notna() &
            (df_dedup["grossprofit_margin"] > df_dedup["gpm_p"])
        ).astype(int)
        df_dedup["f9"] = (
            df_dedup["assets_turn"].notna() & df_dedup["at_p"].notna() &
            (df_dedup["assets_turn"] > df_dedup["at_p"])
        ).astype(int)

        df_dedup["fscore"] = (
            df_dedup[["f1","f2","f3","f4","f5","f6","f7","f8","f9"]].sum(axis=1)
        )
        df_dedup["period_months"] = df_dedup["end_date"].str[4:6].map(
            {"03": 3, "06": 6, "09": 9, "12": 12}
        ).fillna(12).astype(int)

        result = df[["ts_code", "ann_date", "end_date", "roe", "eps"]].merge(
            df_dedup[["ts_code", "end_date", "fscore", "period_months"]],
            on=["ts_code", "end_date"],
            how="left",
        )
        result = result.dropna(subset=["ann_date"])
        n_valid = result["fscore"].notna().sum()
        print(
            f"  F-Score 面板: {len(result):,} 条  |  "
            f"{result['ts_code'].nunique()} 只股票  |  有效F-Score: {n_valid:,}"
        )
        return result

    # ------------------------------------------------------------------
    # 市值（总股本）数据
    # ------------------------------------------------------------------

    def _load_total_share(self, conn) -> None:
        """从 daily_basic 获取最新总股本（万股），用于市值估算。"""
        df = conn.execute("""
            WITH ranked AS (
                SELECT ts_code, total_share, total_mv,
                       ROW_NUMBER() OVER (
                           PARTITION BY ts_code ORDER BY trade_date DESC
                       ) AS rn
                FROM daily_basic
                WHERE total_share > 0
            )
            SELECT ts_code, total_share, total_mv
            FROM ranked WHERE rn = 1
        """).fetchdf()
        self._total_share = dict(zip(df["ts_code"], df["total_share"]))
        print(f"  总股本数据: {len(self._total_share)} 只股票")

    def _load_daily_basic_matrices(self, conn) -> None:
        """
        从 daily_basic 加载历史市值（total_mv，万元→亿元）和 PE TTM 矩阵。
        覆盖 2016-2026，精度优于"最新总股本×当日价格"近似。
        """
        print("  [4/4] 加载 daily_basic 历史市值 + PE TTM 矩阵...")
        df = conn.execute("""
            SELECT trade_date, ts_code,
                   total_mv / 10000.0 AS mktcap,   -- 万元 → 亿元
                   pe_ttm
            FROM daily_basic
            WHERE trade_date >= '20200101'
              AND total_mv > 0
            ORDER BY trade_date, ts_code
        """).fetchdf()

        if df.empty:
            print("  ⚠ daily_basic 无数据，使用总股本估算市值")
            return

        self._mktcap_matrix = df.pivot(index="trade_date", columns="ts_code", values="mktcap")
        self._pe_matrix     = df.pivot(index="trade_date", columns="ts_code", values="pe_ttm")
        print(
            f"  历史市值/PE 矩阵: {self._mktcap_matrix.shape[0]} 天 × "
            f"{self._mktcap_matrix.shape[1]} 只股票"
        )

    def _get_mktcap(self, ts_code: str, close: float, date: str = "") -> float:
        """
        返回市值（亿元）。
        优先使用 daily_basic.total_mv 历史矩阵；无数据时回退到总股本估算。
        """
        if self._mktcap_matrix is not None and date:
            if date in self._mktcap_matrix.index and ts_code in self._mktcap_matrix.columns:
                val = self._mktcap_matrix.at[date, ts_code]
                if not np.isnan(val):
                    return float(val)
        if self._total_share is None:
            return float("nan")
        shares = self._total_share.get(ts_code)
        if shares is None or np.isnan(shares) or shares <= 0:
            return float("nan")
        return shares * close / 10000.0  # 万股 × 元 / 10000 = 亿元

    # ------------------------------------------------------------------
    # 资金流向评分矩阵
    # ------------------------------------------------------------------

    def _load_moneyflow_matrix(self, conn) -> None:
        """
        加载 moneyflow.net_mf_amount，计算 5 日滚动净流入评分（横截面 rank → [0,1]）。
        """
        print("  [3/3] 加载资金流向数据...")
        df = conn.execute("""
            SELECT trade_date, ts_code, net_mf_amount
            FROM moneyflow
            WHERE trade_date >= '20200101'
            ORDER BY trade_date, ts_code
        """).fetchdf()

        if df.empty:
            print("  ⚠ moneyflow 为空，资金面评分将为 0.5")
            return

        mat = df.pivot(index="trade_date", columns="ts_code", values="net_mf_amount")
        roll5 = mat.rolling(window=5, min_periods=1).sum()
        self._mf_score_matrix = roll5.rank(axis=1, pct=True, ascending=True)
        print(
            f"  资金流向矩阵: {self._mf_score_matrix.shape[0]} 天 × "
            f"{self._mf_score_matrix.shape[1]} 只股票"
        )

    def _get_mf_score(self, ts_code: str, date: str) -> float:
        """返回当日资金流向评分 [0,1]，无数据时返回 0.5（中性）。"""
        if self._mf_score_matrix is None:
            return 0.5
        if date not in self._mf_score_matrix.index:
            return 0.5
        if ts_code not in self._mf_score_matrix.columns:
            return 0.5
        val = self._mf_score_matrix.at[date, ts_code]
        return 0.5 if pd.isna(val) else float(val)

    # ------------------------------------------------------------------
    # 时序基本面查询（二分查找）
    # ------------------------------------------------------------------

    def _get_latest_fundamentals(self, date: str) -> Dict[str, Tuple[float, float, float, int]]:
        """截至 date（含）最新披露的 (fscore, roe, eps, period_months)。"""
        if self._fundamental_sorted is None:
            return {}
        result: Dict[str, Tuple[float, float, float, int]] = {}
        for ts_code, arrays in self._fundamental_sorted.items():
            dates = arrays["dates"]
            idx = int(np.searchsorted(dates, date, side="right")) - 1
            if idx >= 0:
                fscore = float(arrays["fscores"][idx])
                roe    = float(arrays["roes"][idx])
                if not (np.isnan(fscore) or np.isnan(roe)):
                    eps = float(arrays["eps"][idx])
                    pm  = int(arrays["period_months"][idx])
                    result[ts_code] = (fscore, roe, eps, pm)
        return result

    # ------------------------------------------------------------------
    # 市场指数仓位控制
    # ------------------------------------------------------------------

    def set_market_index(
        self,
        index_data: pd.DataFrame,
        max_position_pct: float = 0.5,
        ma_period: int = 20,
        ma_long_period: int = 60,
    ) -> None:
        self._index_max_pos = max_position_pct
        idx: pd.Series = index_data.set_index("trade_date")["close"]
        self._index_series = idx
        self._index_ma20_series = idx.rolling(window=ma_period, min_periods=ma_period).mean()
        self._index_ma60_series = idx.rolling(window=ma_long_period, min_periods=ma_long_period).mean()
        print(
            f"✓ 市场指数已设置: {len(idx)} 个交易日  "
            f"三档仓位: SSE>MA{ma_long_period}→100%  MA{ma_period}<SSE≤MA{ma_long_period}→50%  SSE<MA{ma_period}→停止新增"
        )

    def _get_market_ratio(self, date: str) -> float:
        """
        三档市场状态（基于价格相对位置）：
          SSE >= MA60 → 1.0（牛市，满仓）
          MA20 <= SSE < MA60 → 0.5（震荡，半仓）
          SSE < MA20 → 0.0（熊市，停止新增仓位）
        """
        if self._index_series is None or self._index_ma20_series is None:
            return 1.0
        raw_price = self._index_series.get(date)
        raw_ma20  = self._index_ma20_series.get(date)
        if raw_price is None or raw_ma20 is None:
            return 1.0
        price = float(raw_price)  # type: ignore[arg-type]
        ma20  = float(raw_ma20)   # type: ignore[arg-type]
        if pd.isna(price) or pd.isna(ma20):
            return 1.0

        if self._index_ma60_series is not None:
            raw_ma60 = self._index_ma60_series.get(date)
            if raw_ma60 is not None:
                ma60 = float(raw_ma60)  # type: ignore[arg-type]
                if not pd.isna(ma60):
                    if price < ma20:
                        return 0.0   # 熊市：停止新增
                    elif price < ma60:
                        return 0.5   # 震荡：半仓
                    else:
                        return 1.0   # 牛市：满仓

        # 降级为两档
        return self._index_max_pos if price < ma20 else 1.0

    # ------------------------------------------------------------------
    # 矩阵预计算
    # ------------------------------------------------------------------

    def prepare_ma_matrices(self) -> None:
        if self._ma_short_matrix is not None:
            return
        if self.price_matrix is None:
            return
        self._ma_short_matrix = self.price_matrix.rolling(
            window=self.ma_short, min_periods=self.ma_short
        ).mean()
        self._ma_long_matrix = self.price_matrix.rolling(
            window=self.ma_long, min_periods=self.ma_long
        ).mean()
        self._ma_mid_matrix = self.price_matrix.rolling(
            window=60, min_periods=30
        ).mean()
        print(f"✓ MA矩阵缓存完成: MA{self.ma_short} / MA{self.ma_long} / MA60")

    def prepare_vol_matrix(self, data: pd.DataFrame) -> None:
        if self.vol_matrix is not None:
            return
        if "vol" not in data.columns:
            return
        self.vol_matrix = data.pivot(
            index="trade_date", columns="ts_code", values="vol"
        )
        print("✓ 成交量矩阵缓存完成")

    def prepare_rsi_matrix(self) -> None:
        if self._rsi_matrix is not None:
            return
        if self.price_matrix is None:
            return
        delta    = self.price_matrix.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        self._rsi_matrix = 100.0 - 100.0 / (1.0 + rs)
        print(f"✓ RSI矩阵缓存完成: RSI({self.rsi_period})")

    def prepare_vol_ratio_matrix(self) -> None:
        if self._vol_ratio_matrix is not None:
            return
        if self.vol_matrix is None:
            return
        avg_vol = self.vol_matrix.shift(1).rolling(
            window=self.vol_ratio_period, min_periods=self.vol_ratio_period
        ).mean()
        self._vol_ratio_matrix = self.vol_matrix / avg_vol
        print(f"✓ 量比矩阵缓存完成: 基准{self.vol_ratio_period}日")

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_float_dict(row: Union[pd.Series, pd.DataFrame]) -> Dict[str, float]:
        s: pd.Series = row.squeeze() if isinstance(row, pd.DataFrame) else row  # type: ignore
        result: Dict[str, float] = {}
        for k, v in s.items():
            if not pd.isna(v):
                result[str(k)] = float(v)  # type: ignore[arg-type]
        return result

    def _scalar(self, matrix: Optional[pd.DataFrame], ts_code: str, date: str) -> Optional[float]:
        """从预计算矩阵取单个值，返回 None 表示无数据。"""
        if matrix is None or date not in matrix.index or ts_code not in matrix.columns:
            return None
        v = matrix.at[date, ts_code]
        return float(v) if not pd.isna(v) else None

    # ------------------------------------------------------------------
    # 综合评分
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        ts_code: str,
        date: str,
        fscore: float,
        roe: float,
        pe_pct: float,
    ) -> float:
        """
        综合评分 = 0.40 × 基本面 + 0.30 × 技术面 + 0.30 × 资金面
        各子项归一化到 [0, 1]。
        """
        # ---- 基本面 ----
        s_fscore = fscore / 9.0
        s_roe    = min(roe / 30.0, 1.0) if roe > 0 else 0.0
        s_pe     = max(0.0, 1.0 - pe_pct / 100.0)
        fundamental = (
            self._w["fscore"] * s_fscore +
            self._w["roe"]    * s_roe +
            self._w["pe"]     * s_pe
        )

        # ---- 技术面 ----
        rsi     = self._scalar(self._rsi_matrix, ts_code, date)
        ma5     = self._scalar(self._ma_short_matrix, ts_code, date)
        ma20    = self._scalar(self._ma_long_matrix, ts_code, date)
        ma60    = self._scalar(self._ma_mid_matrix, ts_code, date)
        vr      = self._scalar(self._vol_ratio_matrix, ts_code, date)
        close   = self._scalar(self.price_matrix, ts_code, date) if self.price_matrix is not None else None

        s_rsi = max(0.0, (self.rsi_threshold - rsi) / self.rsi_threshold) if rsi is not None else 0.0
        s_ma  = (
            min(max((ma5 / ma20 - 1.0) / 0.10, 0.0), 1.0)
            if ma5 is not None and ma20 is not None and ma20 > 0
            else 0.0
        )
        s_mom = (
            min(max((close / ma60 - 1.0) / 0.15, 0.0), 1.0)
            if close is not None and ma60 is not None and ma60 > 0
            else 0.0
        )
        technical = (
            self._w["rsi"] * s_rsi +
            self._w["ma"]  * s_ma  +
            self._w["mom"] * s_mom
        )

        # ---- 资金面 ----
        s_vr = min(vr / 5.0, 1.0) if vr is not None else 0.0
        s_mf = self._get_mf_score(ts_code, date)
        capital = (
            self._w["volratio"] * s_vr +
            self._w["mf"]       * s_mf
        )

        return 0.40 * fundamental + 0.30 * technical + 0.30 * capital

    # ------------------------------------------------------------------
    # 买入候选筛选
    # ------------------------------------------------------------------

    def _find_entry_candidates(
        self,
        date: str,
        prices: Dict[str, float],
        fund_data: Dict[str, Tuple[float, float, float, int]],
        pe_pct_series: pd.Series,
    ) -> Dict[str, float]:
        """
        返回满足全部买入条件的候选股票 → 综合评分 Dict。
        买入条件：F-Score + ROE + mktcap + PE + MA5>MA20 + RSI<60 + 量比>1.5
        """
        if (self._ma_short_matrix is None or self._ma_long_matrix is None
                or self._rsi_matrix is None or self._vol_ratio_matrix is None):
            return {}
        if date not in (self._ma_short_matrix.index if self._ma_short_matrix is not None else []):
            return {}

        ma5_row  = cast(pd.Series, self._ma_short_matrix.loc[date])
        ma20_row = cast(pd.Series, self._ma_long_matrix.loc[date])
        rsi_row  = cast(pd.Series, self._rsi_matrix.loc[date])
        vr_row   = cast(pd.Series, self._vol_ratio_matrix.loc[date])

        candidates: Dict[str, float] = {}
        for ts_code, (fscore, roe, eps, months) in fund_data.items():
            if ts_code not in prices:
                continue
            close = prices[ts_code]

            if fscore < self.min_fscore:
                continue
            if roe < self.min_roe:
                continue

            mktcap = self._get_mktcap(ts_code, close, date)
            if not np.isnan(mktcap):
                if mktcap < self.min_mktcap or mktcap > self.max_mktcap:
                    continue

            pe_pct = float(pe_pct_series.get(ts_code, float("nan")))  # type: ignore
            if np.isnan(pe_pct) or pe_pct >= self.max_pe_percentile:
                continue

            ma5  = ma5_row.get(ts_code)
            ma20 = ma20_row.get(ts_code)
            if ma5 is None or ma20 is None or pd.isna(ma5) or pd.isna(ma20):
                continue
            if ma5 <= ma20:
                continue

            rsi = rsi_row.get(ts_code)
            if rsi is None or pd.isna(rsi) or rsi >= self.rsi_threshold:
                continue

            vr = vr_row.get(ts_code)
            if vr is None or pd.isna(vr) or vr < self.vol_ratio_threshold:
                continue

            score = self._compute_score(ts_code, date, fscore, roe, pe_pct)
            candidates[ts_code] = score

        return candidates

    # ------------------------------------------------------------------
    # 持有条件检查（每日，不含 RSI/量比/PE）
    # ------------------------------------------------------------------

    def _check_hold_conditions(
        self,
        date: str,
        prices: Dict[str, float],
        current_positions: Set[str],
        fund_data: Dict[str, Tuple[float, float, float, int]],
        trade_dates: list,
        min_hold_days: int = 5,
        ma_cross_days: int = 2,
    ) -> Set[str]:
        """
        返回需要清仓的股票集合。
        持有条件（任一违反则清仓）：
          - MA5 连续 ma_cross_days 日死叉才触发（减少假死叉）
          - 最高点回撤 >= stop_loss_pct（立即触发，不受最短持有期限制）
          - 价格 < entry_price × (1 - cost_stop_loss_pct)（立即触发）
          - F-Score < min_fscore  OR  ROE < min_roe（基本面恶化）
        最短持有期：
          - 买入后 min_hold_days 个交易日内不触发 MA/基本面出场
        """
        to_exit: Set[str] = set()

        if (self._ma_short_matrix is None or self._ma_long_matrix is None
                or date not in self._ma_short_matrix.index):
            return to_exit

        ma5_row  = cast(pd.Series, self._ma_short_matrix.loc[date])
        ma20_row = cast(pd.Series, self._ma_long_matrix.loc[date])

        try:
            date_idx = trade_dates.index(date)
        except ValueError:
            date_idx = len(trade_dates) - 1

        for ts_code in current_positions:
            price = prices.get(ts_code)
            if price is None:
                continue

            self.holding_high[ts_code] = max(
                self.holding_high.get(ts_code, price), price
            )

            # --- 硬止损（不受最短持有期限制）---
            high = self.holding_high.get(ts_code, price)
            if high > 0 and (high - price) / high >= self.stop_loss_pct:
                to_exit.add(ts_code)
                print(
                    f"  [止损-回撤] {ts_code} 最高={high:.2f} 当前={price:.2f} "
                    f"回撤={(high-price)/high:.1%}"
                )
                continue

            entry = self.entry_prices.get(ts_code)
            if entry is not None and price < entry * (1.0 - self.cost_stop_loss_pct):
                to_exit.add(ts_code)
                print(
                    f"  [止损-成本] {ts_code} 成本={entry:.2f} 当前={price:.2f} "
                    f"亏损={(price/entry-1):.1%}"
                )
                continue

            # --- 检查最短持有期 ---
            entry_date = self.entry_dates.get(ts_code, "")
            held_days = 0
            if entry_date and entry_date in trade_dates:
                try:
                    held_days = date_idx - trade_dates.index(entry_date)
                except ValueError:
                    held_days = min_hold_days

            if held_days < min_hold_days:
                self.ma_cross_count.pop(ts_code, None)
                continue

            # --- 趋势 / 基本面出场（最短持有期后才检查）---
            ma5  = ma5_row.get(ts_code)
            ma20 = ma20_row.get(ts_code)
            if ma5 is not None and ma20 is not None and not pd.isna(ma5) and not pd.isna(ma20):
                if ma5 < ma20:
                    self.ma_cross_count[ts_code] = self.ma_cross_count.get(ts_code, 0) + 1
                    if self.ma_cross_count[ts_code] >= ma_cross_days:
                        to_exit.add(ts_code)
                        continue
                else:
                    self.ma_cross_count.pop(ts_code, None)

            fund = fund_data.get(ts_code)
            if fund is not None:
                fscore, roe, *_ = fund
                if fscore < self.min_fscore or roe < self.min_roe:
                    to_exit.add(ts_code)
                    continue

        return to_exit

    # ------------------------------------------------------------------
    # 主循环：每日执行
    # ------------------------------------------------------------------

    def on_bar(self, data: pd.DataFrame, date: str) -> Dict[str, float]:
        """
        每日执行：
          1. 预计算矩阵（仅首次）
          2. 检查持有条件 → 清仓违规股票
          3. 扫描全市场买入候选 → 综合评分 → 选 top_n
          4. 仅在持仓变化时返回新权重，否则返回 {}
        """
        self.prepare_price_matrix(data)
        self.prepare_vol_matrix(data)

        pm = self.price_matrix
        assert pm is not None

        if not self.trade_dates or date not in self.trade_dates:
            if not hasattr(self, "_all_trade_dates"):
                self._all_trade_dates = pm.index.tolist()
            self.trade_dates = [d for d in self._all_trade_dates if d <= date]

        if self._fundamental_sorted is None:
            return {}
        if date not in pm.index:
            return {}

        self.prepare_ma_matrices()
        self.prepare_rsi_matrix()
        self.prepare_vol_ratio_matrix()

        prices = self._row_to_float_dict(cast(pd.Series, pm.loc[date]))
        fund_data = self._get_latest_fundamentals(date)

        # ---- PE 横截面百分位（优先 pe_ttm，回退到年化 EPS）----
        pe_combined: Dict[str, float] = {}
        if self._pe_matrix is not None and date in self._pe_matrix.index:
            pe_row = cast(pd.Series, self._pe_matrix.loc[date])
            for ts_code_raw, val in pe_row.items():
                if 0 < val < 500:
                    pe_combined[str(ts_code_raw)] = float(val)

        for ts_code, (_, _, eps, months) in fund_data.items():
            if ts_code in pe_combined:
                continue
            p = prices.get(ts_code)
            if p is None or np.isnan(eps) or eps <= 0 or months <= 0:
                continue
            ann_eps = eps * (12.0 / months)
            if ann_eps <= 0:
                continue
            pe = p / ann_eps
            if 0 < pe < 500:
                pe_combined[ts_code] = pe

        pe_pct_series: pd.Series = (
            pd.Series(pe_combined).rank(pct=True, ascending=True) * 100
            if pe_combined else pd.Series(dtype=float)
        )

        # ---- 1. 检查当前持仓的持有条件 ----
        current = set(self.positions.keys())
        to_exit = self._check_hold_conditions(
            date, prices, current, fund_data, self.trade_dates,
            min_hold_days=10, ma_cross_days=3,
        )

        for ts_code in to_exit:
            self.holding_high.pop(ts_code, None)
            self.entry_prices.pop(ts_code, None)
            self.entry_dates.pop(ts_code, None)
            self.ma_cross_count.pop(ts_code, None)

        remaining = {k: v for k, v in self.positions.items() if k not in to_exit}

        # ---- 2. 扫描买入候选 ----
        scored_candidates = self._find_entry_candidates(date, prices, fund_data, pe_pct_series)
        new_candidates = {
            ts: score for ts, score in scored_candidates.items()
            if ts not in remaining
        }

        # ---- 3. 决定新持仓 ----
        slot_weight = 1.0 / self.top_n

        market_ratio = self._get_market_ratio(date)
        if market_ratio != self._last_market_ratio:
            if market_ratio == 0.0:
                print(f"  [仓位控制] {date} 上证 < MA20（熊市），停止新增仓位")
            elif market_ratio < 1.0:
                print(f"  [仓位控制] {date} 上证 MA20~MA60（震荡），槽位上限 {int(self.top_n * market_ratio)}/{self.top_n}")
            else:
                print(f"  [仓位控制] {date} 上证 > MA60（牛市），满仓运行 {self.top_n}/{self.top_n}")
        self._last_market_ratio = market_ratio

        if market_ratio == 0.0:
            max_slots = 0
        elif market_ratio <= 0.5:
            max_slots = int(self.top_n * 0.5)
        else:
            max_slots = self.top_n

        slots = max(max_slots - len(remaining), 0)
        new_entries_sorted = sorted(new_candidates.items(), key=lambda x: x[1], reverse=True)
        new_entry_codes = [ts for ts, _ in new_entries_sorted[:slots]]

        changed = bool(to_exit) or bool(new_entry_codes)
        if not changed:
            return {}

        all_holdings = list(remaining.keys()) + new_entry_codes
        if not all_holdings:
            self.update_positions({})
            return {}

        for ts_code in new_entry_codes:
            p = prices.get(ts_code)
            if p is not None:
                self.holding_high[ts_code] = p
                self.entry_prices[ts_code] = p
                self.entry_dates[ts_code] = date
                self.ma_cross_count.pop(ts_code, None)

        new_positions = {ts: slot_weight for ts in all_holdings}
        self.update_positions(new_positions)
        return new_positions

    # ------------------------------------------------------------------
    # 策略信息
    # ------------------------------------------------------------------

    def get_strategy_info(self) -> Dict:
        return {
            "name": self.name,
            "top_n": self.top_n,
            "min_fscore": self.min_fscore,
            "max_pe_percentile": self.max_pe_percentile,
            "min_roe": self.min_roe,
            "mktcap_range": f"{self.min_mktcap}亿 – {self.max_mktcap}亿",
            "buy_conditions": [
                f"市值 {self.min_mktcap}亿–{self.max_mktcap}亿",
                f"F-Score >= {self.min_fscore}（9维Piotroski同比）",
                f"ROE > {self.min_roe}%",
                f"MA{self.ma_short} > MA{self.ma_long}（趋势向上）",
                f"RSI({self.rsi_period}) < {self.rsi_threshold}（买入条件，持有不检查）",
                f"量比 > {self.vol_ratio_threshold}（买入条件，持有不检查）",
                f"PE横截面百分位 < {self.max_pe_percentile}%（买入条件，持有不检查）",
            ],
            "hold_conditions": [
                f"F-Score >= {self.min_fscore}  AND  ROE > {self.min_roe}%",
                f"MA{self.ma_short} 连续死叉 < 3 天",
                f"最高点回撤 < {self.stop_loss_pct:.0%}",
                f"价格 > 成本价 × {1-self.cost_stop_loss_pct:.0%}",
            ],
            "scoring": "基本面 40% + 技术面 30% + 资金面 30%",
            "position_control": (
                f"固定槽位权重 1/{self.top_n}；三档市场仓位（牛/震荡/熊）"
                if self._index_series is not None
                else f"固定槽位权重 1/{self.top_n}"
            ),
        }
