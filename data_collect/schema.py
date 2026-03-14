"""
DuckDB 表结构定义
"""

# 股票基本信息表
STOCK_BASIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_basic (
    ts_code VARCHAR PRIMARY KEY,      -- 股票代码（带市场后缀）
    symbol VARCHAR NOT NULL,          -- 股票代码（不带后缀）
    name VARCHAR NOT NULL,            -- 股票名称
    area VARCHAR,                     -- 地区
    industry VARCHAR,                 -- 行业
    list_date VARCHAR,                -- 上市日期
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 更新时间
);
CREATE INDEX IF NOT EXISTS idx_stock_basic_industry ON stock_basic(industry);
CREATE INDEX IF NOT EXISTS idx_stock_basic_area ON stock_basic(area);
"""

# 日线行情表（前复权）
DAILY_PRICE_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_price (
    ts_code VARCHAR NOT NULL,         -- 股票代码
    trade_date VARCHAR NOT NULL,      -- 交易日期
    open DOUBLE,                      -- 开盘价（前复权）
    high DOUBLE,                      -- 最高价（前复权）
    low DOUBLE,                       -- 最低价（前复权）
    close DOUBLE,                     -- 收盘价（前复权）
    pre_close DOUBLE,                 -- 昨收价（前复权）
    change DOUBLE,                    -- 涨跌额
    pct_chg DOUBLE,                   -- 涨跌幅
    vol DOUBLE,                       -- 成交量（手）
    amount DOUBLE,                    -- 成交额（千元）
    adj_factor DOUBLE,                -- 复权因子
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_price_date ON daily_price(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_price_code ON daily_price(ts_code);
"""

# 资产负债表
BALANCE_SHEET_SCHEMA = """
CREATE TABLE IF NOT EXISTS balance_sheet (
    ts_code VARCHAR NOT NULL,         -- 股票代码
    end_date VARCHAR NOT NULL,        -- 报告期
    ann_date VARCHAR,                 -- 公告日期
    f_ann_date VARCHAR,               -- 实际公告日期
    report_type VARCHAR,              -- 报告类型
    comp_type VARCHAR,                -- 公司类型
    total_assets DOUBLE,              -- 资产总计
    total_liab DOUBLE,                -- 负债合计
    total_hldr_eqy_inc_min_int DOUBLE, -- 股东权益合计（含少数股东权益）
    total_cur_assets DOUBLE,          -- 流动资产合计
    total_nca DOUBLE,                 -- 非流动资产合计
    total_cur_liab DOUBLE,            -- 流动负债合计
    total_ncl DOUBLE,                 -- 非流动负债合计
    money_cap DOUBLE,                 -- 货币资金
    accounts_receiv DOUBLE,           -- 应收账款
    inventories DOUBLE,               -- 存货
    fix_assets DOUBLE,                -- 固定资产
    accounts_pay DOUBLE,              -- 应付账款
    st_borr DOUBLE,                   -- 短期借款
    lt_borr DOUBLE,                   -- 长期借款
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, end_date, ann_date)
);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_date ON balance_sheet(end_date);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_code ON balance_sheet(ts_code);
"""

# 利润表
INCOME_STATEMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS income_statement (
    ts_code VARCHAR NOT NULL,         -- 股票代码
    end_date VARCHAR NOT NULL,        -- 报告期
    ann_date VARCHAR,                 -- 公告日期
    f_ann_date VARCHAR,               -- 实际公告日期
    report_type VARCHAR,              -- 报告类型
    comp_type VARCHAR,                -- 公司类型
    basic_eps DOUBLE,                 -- 基本每股收益
    diluted_eps DOUBLE,               -- 稀释每股收益
    total_revenue DOUBLE,             -- 营业总收入
    revenue DOUBLE,                   -- 营业收入
    operate_profit DOUBLE,            -- 营业利润
    total_profit DOUBLE,              -- 利润总额
    n_income DOUBLE,                  -- 净利润
    n_income_attr_p DOUBLE,           -- 归属于母公司所有者的净利润
    oper_cost DOUBLE,                 -- 营业成本
    sell_exp DOUBLE,                  -- 销售费用
    admin_exp DOUBLE,                 -- 管理费用
    fin_exp DOUBLE,                   -- 财务费用
    rd_exp DOUBLE,                    -- 研发费用
    ebit DOUBLE,                      -- 息税前利润
    ebitda DOUBLE,                    -- 息税折旧摊销前利润
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, end_date, ann_date)
);
CREATE INDEX IF NOT EXISTS idx_income_statement_date ON income_statement(end_date);
CREATE INDEX IF NOT EXISTS idx_income_statement_code ON income_statement(ts_code);
"""

# 现金流量表
CASH_FLOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS cash_flow (
    ts_code VARCHAR NOT NULL,         -- 股票代码
    end_date VARCHAR NOT NULL,        -- 报告期
    ann_date VARCHAR,                 -- 公告日期
    f_ann_date VARCHAR,               -- 实际公告日期
    report_type VARCHAR,              -- 报告类型
    comp_type VARCHAR,                -- 公司类型
    n_cashflow_act DOUBLE,            -- 经营活动产生的现金流量净额
    n_cashflow_inv_act DOUBLE,        -- 投资活动产生的现金流量净额
    n_cash_flows_fnc_act DOUBLE,      -- 筹资活动产生的现金流量净额
    c_cash_equ_end_period DOUBLE,     -- 期末现金及现金等价物余额
    c_cash_equ_beg_period DOUBLE,     -- 期初现金及现金等价物余额
    n_incr_cash_cash_equ DOUBLE,      -- 现金及现金等价物净增加额
    free_cashflow DOUBLE,             -- 企业自由现金流量
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, end_date, ann_date)
);
CREATE INDEX IF NOT EXISTS idx_cash_flow_date ON cash_flow(end_date);
CREATE INDEX IF NOT EXISTS idx_cash_flow_code ON cash_flow(ts_code);
"""

# 数据更新日志表
UPDATE_LOG_SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS update_log_seq START 1;
CREATE TABLE IF NOT EXISTS update_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('update_log_seq'),
    table_name VARCHAR NOT NULL,      -- 表名
    update_type VARCHAR NOT NULL,     -- 更新类型：full/incremental
    start_date VARCHAR,               -- 更新起始日期
    end_date VARCHAR,                 -- 更新结束日期
    records_count INTEGER,            -- 更新记录数
    status VARCHAR NOT NULL,          -- 状态：success/failed
    error_message TEXT,               -- 错误信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_update_log_table ON update_log(table_name);
CREATE INDEX IF NOT EXISTS idx_update_log_date ON update_log(created_at);
"""

# 每日指标表（市值、估值等）
DAILY_BASIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_basic (
    ts_code VARCHAR NOT NULL,         -- 股票代码
    trade_date VARCHAR NOT NULL,      -- 交易日期
    close DOUBLE,                     -- 收盘价
    turnover_rate DOUBLE,             -- 换手率（%）
    turnover_rate_f DOUBLE,           -- 换手率（自由流通股）
    volume_ratio DOUBLE,              -- 量比
    pe DOUBLE,                        -- 市盈率（总市值/净利润）
    pe_ttm DOUBLE,                    -- 市盈率TTM
    pb DOUBLE,                        -- 市净率
    ps DOUBLE,                        -- 市销率
    ps_ttm DOUBLE,                    -- 市销率TTM
    dv_ratio DOUBLE,                  -- 股息率（%）
    dv_ttm DOUBLE,                    -- 股息率TTM（%）
    total_share DOUBLE,               -- 总股本（万股）
    float_share DOUBLE,               -- 流通股本（万股）
    free_share DOUBLE,                -- 自由流通股本（万股）
    total_mv DOUBLE,                  -- 总市值（万元）
    circ_mv DOUBLE,                   -- 流通市值（万元）
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_daily_basic_date ON daily_basic(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_basic_code ON daily_basic(ts_code);
CREATE INDEX IF NOT EXISTS idx_daily_basic_mv ON daily_basic(total_mv);
"""

# 财务指标表（ROE、ROA等）
FINA_INDICATOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS fina_indicator (
    ts_code VARCHAR NOT NULL,         -- 股票代码
    ann_date VARCHAR NOT NULL,        -- 公告日期
    end_date VARCHAR NOT NULL,        -- 报告期
    -- 盈利能力指标
    roe DOUBLE,                       -- 净资产收益率（%）
    roe_waa DOUBLE,                   -- 加权平均净资产收益率（%）
    roe_dt DOUBLE,                    -- 净资产收益率-扣除非经常损益（%）
    roa DOUBLE,                       -- 总资产报酬率（%）
    roic DOUBLE,                      -- 投资资本回报率（%）
    gross_margin DOUBLE,              -- 销售毛利率（%）
    netprofit_margin DOUBLE,          -- 销售净利率（%）
    grossprofit_margin DOUBLE,        -- 毛利率（%）
    -- 营运能力指标
    assets_turn DOUBLE,               -- 总资产周转率（次）
    ar_turn DOUBLE,                   -- 应收账款周转率（次）
    ca_turn DOUBLE,                   -- 流动资产周转率（次）
    fa_turn DOUBLE,                   -- 固定资产周转率（次）
    turn_days DOUBLE,                 -- 存货周转天数（天）
    -- 偿债能力指标
    current_ratio DOUBLE,             -- 流动比率
    quick_ratio DOUBLE,               -- 速动比率
    cash_ratio DOUBLE,                -- 现金比率
    debt_to_assets DOUBLE,            -- 资产负债率（%）
    debt_to_eqt DOUBLE,               -- 产权比率
    eqt_to_debt DOUBLE,               -- 权益乘数
    -- 成长能力指标
    basic_eps_yoy DOUBLE,             -- 基本每股收益同比增长率（%）
    netprofit_yoy DOUBLE,             -- 净利润同比增长率（%）
    op_yoy DOUBLE,                    -- 营业利润同比增长率（%）
    roe_yoy DOUBLE,                   -- ROE同比增长率（%）
    tr_yoy DOUBLE,                    -- 营业总收入同比增长率（%）
    or_yoy DOUBLE,                    -- 营业收入同比增长率（%）
    -- 每股指标
    eps DOUBLE,                       -- 基本每股收益
    dt_eps DOUBLE,                    -- 稀释每股收益
    bps DOUBLE,                       -- 每股净资产
    ocfps DOUBLE,                     -- 每股经营现金流
    cfps DOUBLE,                      -- 每股现金流
    -- 其他重要指标
    ebit DOUBLE,                      -- 息税前利润
    ebitda DOUBLE,                    -- 息税折旧摊销前利润
    fcff DOUBLE,                      -- 企业自由现金流
    fcfe DOUBLE,                      -- 股权自由现金流
    profit_dedt DOUBLE,               -- 扣除非经常损益后的净利润
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, end_date, ann_date)
);
CREATE INDEX IF NOT EXISTS idx_fina_indicator_date ON fina_indicator(end_date);
CREATE INDEX IF NOT EXISTS idx_fina_indicator_code ON fina_indicator(ts_code);
CREATE INDEX IF NOT EXISTS idx_fina_indicator_roe ON fina_indicator(roe);
"""

# 资金流向表
MONEYFLOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS moneyflow (
    ts_code VARCHAR NOT NULL,         -- 股票代码
    trade_date VARCHAR NOT NULL,      -- 交易日期
    buy_sm_amount DOUBLE,             -- 小单买入金额（万元）
    sell_sm_amount DOUBLE,            -- 小单卖出金额（万元）
    buy_md_amount DOUBLE,             -- 中单买入金额（万元）
    sell_md_amount DOUBLE,            -- 中单卖出金额（万元）
    buy_lg_amount DOUBLE,             -- 大单买入金额（万元）
    sell_lg_amount DOUBLE,            -- 大单卖出金额（万元）
    buy_elg_amount DOUBLE,            -- 特大单买入金额（万元）
    sell_elg_amount DOUBLE,           -- 特大单卖出金额（万元）
    net_mf_amount DOUBLE,             -- 净流入金额（万元）
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_moneyflow_date ON moneyflow(trade_date);
CREATE INDEX IF NOT EXISTS idx_moneyflow_code ON moneyflow(ts_code);
"""

# 指数基本信息表
INDEX_BASIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_basic (
    ts_code VARCHAR PRIMARY KEY,      -- 指数代码
    name VARCHAR NOT NULL,            -- 指数名称
    market VARCHAR,                   -- 市场
    publisher VARCHAR,                -- 发布方
    category VARCHAR,                 -- 指数类别
    base_date VARCHAR,                -- 基期
    base_point DOUBLE,                -- 基点
    list_date VARCHAR,                -- 发布日期
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_index_basic_market ON index_basic(market);
"""

# 指数日线表
INDEX_DAILY_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_daily (
    ts_code VARCHAR NOT NULL,         -- 指数代码
    trade_date VARCHAR NOT NULL,      -- 交易日期
    open DOUBLE,                      -- 开盘点位
    high DOUBLE,                      -- 最高点位
    low DOUBLE,                       -- 最低点位
    close DOUBLE,                     -- 收盘点位
    pre_close DOUBLE,                 -- 昨收盘点位
    change DOUBLE,                    -- 涨跌点
    pct_chg DOUBLE,                   -- 涨跌幅（%）
    vol DOUBLE,                       -- 成交量（手）
    amount DOUBLE,                    -- 成交额（千元）
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_index_daily_code ON index_daily(ts_code);
"""

# 分红送股表
DIVIDEND_SCHEMA = """
CREATE TABLE IF NOT EXISTS dividend (
    ts_code VARCHAR NOT NULL,
    end_date VARCHAR NOT NULL,
    ann_date VARCHAR,
    div_proc VARCHAR,
    stk_div DOUBLE,
    stk_bo_rate DOUBLE,
    stk_co_rate DOUBLE,
    cash_div DOUBLE,
    cash_div_tax DOUBLE,
    record_date VARCHAR,
    ex_date VARCHAR,
    pay_date VARCHAR,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, end_date)
);
CREATE INDEX IF NOT EXISTS idx_dividend_date ON dividend(end_date);
CREATE INDEX IF NOT EXISTS idx_dividend_code ON dividend(ts_code);
CREATE INDEX IF NOT EXISTS idx_dividend_ex_date ON dividend(ex_date);
"""

# 券商一致预期表（分析师预测数据）
REPORT_RC_SCHEMA = """
CREATE TABLE IF NOT EXISTS report_rc (
    ts_code VARCHAR NOT NULL,         -- 股票代码
    report_date VARCHAR NOT NULL,     -- 报告发布日期
    quarter VARCHAR NOT NULL,         -- 预测季度（如 2026Q4）
    np DOUBLE,                        -- 预测净利润（万元）
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, report_date, quarter)
);
CREATE INDEX IF NOT EXISTS idx_report_rc_code ON report_rc(ts_code);
CREATE INDEX IF NOT EXISTS idx_report_rc_quarter ON report_rc(quarter);
CREATE INDEX IF NOT EXISTS idx_report_rc_date ON report_rc(report_date);
"""

ALL_SCHEMAS = [
    STOCK_BASIC_SCHEMA,
    DAILY_PRICE_SCHEMA,
    BALANCE_SHEET_SCHEMA,
    INCOME_STATEMENT_SCHEMA,
    CASH_FLOW_SCHEMA,
    UPDATE_LOG_SCHEMA,
    DAILY_BASIC_SCHEMA,
    FINA_INDICATOR_SCHEMA,
    MONEYFLOW_SCHEMA,
    INDEX_BASIC_SCHEMA,
    INDEX_DAILY_SCHEMA,
    DIVIDEND_SCHEMA,
    REPORT_RC_SCHEMA
]
