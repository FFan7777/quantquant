#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日数据库增量更新脚本
======================
每天凌晨（A股收盘后，交易日数据落库通常在 18:00 后完成）自动运行，
更新 daily_price / daily_basic / moneyflow / index_daily / 财务报表。

用法:
  python daily_update.py              # 手动运行（跳过非交易日检查）
  python daily_update.py --force      # 强制运行（无论是否交易日）
  python daily_update.py --dry-run    # 只打印计划，不执行

定时配置 (cron，每天 20:00 执行):
  crontab -e
  0 20 * * 1-5 cd /Users/hanenshou/Downloads/quant_claude && /Users/hanenshou/Downloads/quant_claude/venv/bin/python daily_update.py >> logs/daily_update.log 2>&1

macOS launchd (推荐，见脚本末尾的 launchd plist 生成命令)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── 工作目录固定为脚本所在目录（cron 环境下 CWD 可能不对）────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

# ── 日志配置 ──────────────────────────────────────────────────────────────────
LOG_DIR = SCRIPT_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"daily_update_{datetime.now().strftime('%Y%m')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def is_trading_day(date: datetime) -> bool:
    """简单判断是否为交易日（排除周末；节假日需额外维护）"""
    return date.weekday() < 5   # 0=Mon … 4=Fri


def get_latest_db_date(table: str, date_col: str = 'trade_date') -> str:
    """查询指定表的最新数据日期"""
    import duckdb
    from data_collect.config import config
    try:
        with duckdb.connect(config.db_path, read_only=True) as conn:
            result = conn.execute(
                f"SELECT MAX({date_col}) FROM {table}"
            ).fetchone()[0]
        return str(result).replace('-', '') if result else '20160101'
    except Exception as e:
        log.warning(f"查询 {table}.{date_col} 失败: {e}")
        return '20160101'


def retry(func, max_attempts: int = 3, delay: int = 60, label: str = ''):
    """带重试的函数调用（Tushare IP 超限时等待后重试）"""
    for attempt in range(1, max_attempts + 1):
        try:
            func()
            return True
        except Exception as e:
            msg = str(e)
            log.warning(f"[{label}] 第 {attempt}/{max_attempts} 次失败: {msg}")
            if 'IP' in msg or '超限' in msg:
                wait = delay * attempt
                log.info(f"  IP 超限，等待 {wait}s 后重试...")
                time.sleep(wait)
            elif attempt < max_attempts:
                time.sleep(delay)
            else:
                log.error(f"[{label}] 最终失败，跳过")
                return False
    return False


# ── 各步骤更新函数 ────────────────────────────────────────────────────────────

def step_stock_basic(collector) -> bool:
    """更新股票列表（每周一次足够，但增量跑一次无害）"""
    log.info("▶ [1/7] 更新股票基本信息 (stock_basic)...")
    return retry(collector.collect_stock_basic, label='stock_basic')


def step_daily_price(collector) -> bool:
    """增量更新个股日线（前复权）"""
    latest = get_latest_db_date('daily_price')
    log.info(f"▶ [2/7] 增量更新日线价格 (daily_price)，当前最新: {latest}")
    return retry(
        lambda: collector.collect_daily_price(incremental=True),
        label='daily_price',
    )


def step_daily_basic(collector) -> bool:
    """增量更新每日指标（市值/PE/换手率）"""
    latest = get_latest_db_date('daily_basic')
    log.info(f"▶ [3/7] 增量更新每日指标 (daily_basic)，当前最新: {latest}")
    return retry(
        lambda: collector.collect_daily_basic(incremental=True),
        label='daily_basic',
    )


def step_moneyflow(collector) -> bool:
    """增量更新资金流向"""
    latest = get_latest_db_date('moneyflow')
    log.info(f"▶ [4/7] 增量更新资金流向 (moneyflow)，当前最新: {latest}")
    return retry(
        lambda: collector.collect_moneyflow(incremental=True),
        label='moneyflow',
    )


def step_index(collector) -> bool:
    """增量更新指数日线（CSI300 等 6 个主要指数）"""
    latest = get_latest_db_date('index_daily')
    log.info(f"▶ [5/7] 增量更新指数日线 (index_daily)，当前最新: {latest}")
    return retry(
        lambda: collector.collect_index_data(incremental=True),
        label='index_daily',
    )


def step_financial(collector) -> bool:
    """增量更新财务三报表 + 财务指标（最近 90 天，捕获季报新披露）"""
    recent_start = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
    log.info(f"▶ [6/7] 增量更新财务数据（近 90 天，起点 {recent_start}）...")
    ok1 = retry(
        lambda: collector.collect_financial_data(start_date=recent_start),
        label='financial',
        delay=30,
    )
    ok2 = retry(
        lambda: collector.collect_fina_indicator(start_date=recent_start),
        label='fina_indicator',
        delay=30,
    )
    return ok1 and ok2


def step_report_rc(collector) -> bool:
    """增量更新券商一致预期"""
    log.info("▶ [7/7] 增量更新券商一致预期 (report_rc)...")
    return retry(
        lambda: collector.collect_report_rc(incremental=True),
        label='report_rc',
        delay=30,
    )


# ── 汇报最终数据状态 ──────────────────────────────────────────────────────────

def print_summary():
    """打印各关键表最新日期"""
    import duckdb
    from data_collect.config import config

    checks = [
        ('daily_price',      'trade_date'),
        ('daily_basic',      'trade_date'),
        ('moneyflow',        'trade_date'),
        ('index_daily',      'trade_date'),
        ('income_statement', 'f_ann_date'),
        ('fina_indicator',   'ann_date'),
    ]
    log.info("─" * 50)
    log.info("更新后数据库状态:")
    with duckdb.connect(config.db_path) as conn:
        for table, col in checks:
            try:
                r = conn.execute(f"SELECT COUNT(*), MAX({col}) FROM {table}").fetchone()
                log.info(f"  {table:<22}  行数: {r[0]:>10,}  最新: {r[1]}")
            except Exception as e:
                log.warning(f"  {table:<22}  查询失败: {e}")
    log.info("─" * 50)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='每日数据库增量更新')
    parser.add_argument('--force',   action='store_true', help='强制运行（忽略非交易日）')
    parser.add_argument('--dry-run', action='store_true', help='只打印计划，不执行')
    parser.add_argument('--skip-financial', action='store_true',
                        help='跳过财务数据更新（较慢，非季报披露期可跳过）')
    parser.add_argument('--skip-stock-basic', action='store_true',
                        help='跳过 stock_basic 更新（避免 IP 超限时首步报错）')
    args = parser.parse_args()

    now = datetime.now()
    log.info("=" * 60)
    log.info(f"每日数据更新任务启动  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # 非交易日检查（cron 配置 Mon-Fri，但节假日仍会触发）
    if not args.force and not is_trading_day(now):
        log.info(f"今天 {now.strftime('%A')} 是非交易日，跳过更新。（--force 可强制运行）")
        return

    if args.dry_run:
        log.info("[DRY-RUN] 以下步骤将被执行（实际不运行）：")
        log.info("  1. stock_basic  2. daily_price  3. daily_basic")
        log.info("  4. moneyflow    5. index_daily  6. financial   7. report_rc")
        return

    # 初始化 collector
    from data_collect.collector import DataCollector
    collector = DataCollector()

    results = {}

    if not args.skip_stock_basic:
        results['stock_basic'] = step_stock_basic(collector)
        time.sleep(2)

    results['daily_price'] = step_daily_price(collector)
    time.sleep(2)

    results['daily_basic'] = step_daily_basic(collector)
    time.sleep(2)

    results['moneyflow'] = step_moneyflow(collector)
    time.sleep(2)

    results['index_daily'] = step_index(collector)
    time.sleep(2)

    if not args.skip_financial:
        results['financial'] = step_financial(collector)
        time.sleep(2)

    results['report_rc'] = step_report_rc(collector)

    # 汇总
    print_summary()

    failed = [k for k, v in results.items() if not v]
    if failed:
        log.warning(f"以下步骤失败或跳过: {failed}")
    else:
        log.info("所有步骤成功完成 ✓")

    log.info(f"任务结束  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()


# ══════════════════════════════════════════════════════════════════════════════
# macOS launchd 定时任务配置说明
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. 生成 plist 文件（在终端执行）:
#
#    cat > ~/Library/LaunchAgents/com.quant.daily_update.plist << 'EOF'
#    <?xml version="1.0" encoding="UTF-8"?>
#    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
#        "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
#    <plist version="1.0">
#    <dict>
#        <key>Label</key>
#        <string>com.quant.daily_update</string>
#        <key>ProgramArguments</key>
#        <array>
#            <string>/Users/hanenshou/Downloads/quant_claude/venv/bin/python</string>
#            <string>/Users/hanenshou/Downloads/quant_claude/daily_update.py</string>
#        </array>
#        <key>WorkingDirectory</key>
#        <string>/Users/hanenshou/Downloads/quant_claude</string>
#        <key>StartCalendarInterval</key>
#        <dict>
#            <key>Hour</key>
#            <integer>20</integer>
#            <key>Minute</key>
#            <integer>0</integer>
#        </dict>
#        <key>StandardOutPath</key>
#        <string>/Users/hanenshou/Downloads/quant_claude/logs/launchd_out.log</string>
#        <key>StandardErrorPath</key>
#        <string>/Users/hanenshou/Downloads/quant_claude/logs/launchd_err.log</string>
#        <key>RunAtLoad</key>
#        <false/>
#    </dict>
#    </plist>
#    EOF
#
# 2. 加载并启用:
#    launchctl load ~/Library/LaunchAgents/com.quant.daily_update.plist
#
# 3. 验证:
#    launchctl list | grep quant
#
# 4. 手动触发测试:
#    launchctl start com.quant.daily_update
#
# 5. 停用:
#    launchctl unload ~/Library/LaunchAgents/com.quant.daily_update.plist
#
# ─── cron 备选（简单，但 macOS 下 cron 需要完全磁盘访问权限）────────────────
#
#    crontab -e   # 添加下面一行（每天 20:00 周一至周五运行）：
#    0 20 * * 1-5 cd /Users/hanenshou/Downloads/quant_claude && /Users/hanenshou/Downloads/quant_claude/venv/bin/python daily_update.py >> logs/daily_update.log 2>&1
#
