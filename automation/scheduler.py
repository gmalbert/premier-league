"""
Automated Data Pipeline Scheduler
Runs the full data update and model training pipeline on a schedule.

Usage:
    python automation/scheduler.py

Windows Task Scheduler:
    Action: python C:/path/to/premier-league/automation/scheduler.py

Linux cron:
    0 3 * * * cd /path/to/premier-league && python automation/scheduler.py
"""

import schedule
import time
import subprocess
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Ensure we run from the project root regardless of where the script is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/scheduler.log', encoding='utf-8'),
    ]
)
log = logging.getLogger('scheduler')

# ── Pipeline steps ────────────────────────────────────────────────────────────

def _run(module: str, label: str) -> bool:
    """Run an installed core module and return True on success."""
    log.info(f"Starting: {label}")
    result = subprocess.run(
        [sys.executable, '-m', module],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"FAILED: {label}\n{result.stderr}")
        return False
    log.info(f"Completed: {label}")
    return True


def update_historical_data():
    """Download raw CSVs and regenerate processed feature data."""
    log.info("=== Historical data update started ===")
    if not _run('combine_raw_data', 'combine_raw_data'):
        return
    if not _run('prepare_model_data', 'prepare_model_data'):
        return
    log.info("=== Historical data update complete ===")


def retrain_models():
    """Re-train and save all ML models."""
    log.info("=== Model training started ===")
    _run('train_models', 'train_models')
    log.info("=== Model training complete ===")


def update_fixtures():
    """Fetch upcoming Premier League fixtures from ESPN API."""
    log.info("Updating upcoming fixtures...")
    _run('fetch_upcoming_fixtures', 'fetch_upcoming_fixtures')


def scrape_referees():
    """Scrape latest referee assignments from Playmaker Stats."""
    log.info("Scraping referee assignments...")
    _run('scrape_referees', 'scrape_referees')


def precompute_database():
    """Precompute processed data for fast app startup."""
    log.info("Precomputing database...")
    _run('precompute_database', 'precompute_database')


def full_nightly_pipeline():
    """Full overnight refresh: data → models → precompute."""
    log.info("========== NIGHTLY PIPELINE START ==========")
    update_historical_data()
    retrain_models()
    precompute_database()
    update_fixtures()
    scrape_referees()
    log.info("========== NIGHTLY PIPELINE COMPLETE ==========")


# ── Schedule ──────────────────────────────────────────────────────────────────

def setup_schedule():
    # Full nightly run at 3 AM
    schedule.every().day.at("03:00").do(full_nightly_pipeline)

    # Fixtures refresh every hour during the day
    schedule.every().hour.do(update_fixtures)

    # Referee assignments once a week (Monday 6 AM — assignments typically
    # published mid-week for the following weekend)
    schedule.every().monday.at("06:00").do(scrape_referees)

    log.info("Schedule configured:")
    for job in schedule.jobs:
        log.info(f"  {job}")


if __name__ == '__main__':
    log.info(f"Scheduler starting – project root: {PROJECT_ROOT}")
    setup_schedule()

    # Run the full pipeline immediately on first start so data is fresh
    full_nightly_pipeline()

    log.info("Entering schedule loop (Ctrl+C to stop)...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user.")
