import logging
import shutil
from datetime import datetime
from pathlib import Path

LAST_RUN_FILE = Path(".last-run")
LAST_RUN_BACKUP = Path(".last-run.bak")
DATE_FMT = "%d-%b-%Y (%H:%M:%S.%f)"


def set_last_run_date():
    if LAST_RUN_FILE.exists():
        shutil.copy2(LAST_RUN_FILE, LAST_RUN_BACKUP)
    LAST_RUN_FILE.write_text(datetime.today().strftime(DATE_FMT))


def restore_last_run_date():
    """Restore .last-run from .last-run.bak."""
    if LAST_RUN_BACKUP.exists():
        shutil.copy2(LAST_RUN_BACKUP, LAST_RUN_FILE)
        logging.info("Restored .last-run to: %s", LAST_RUN_FILE.read_text())
    else:
        logging.warning("No backup found (.last-run.bak does not exist).")


def get_last_run_date():
    try:
        return datetime.strptime(LAST_RUN_FILE.read_text(), DATE_FMT)
    except (FileNotFoundError, ValueError):
        return datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
