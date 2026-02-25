from datetime import datetime
from pathlib import Path

LAST_RUN_FILE = Path(".last-run")
DATE_FMT = "%d-%b-%Y (%H:%M:%S.%f)"


def set_last_run_date():
    LAST_RUN_FILE.write_text(datetime.today().strftime(DATE_FMT))


def get_last_run_date():
    try:
        return datetime.strptime(LAST_RUN_FILE.read_text(), DATE_FMT)
    except (FileNotFoundError, ValueError):
        return datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
