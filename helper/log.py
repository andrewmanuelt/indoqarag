import json
from datetime import datetime
from pathlib import Path

_LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "log"


def log_llm_request(suffix, record):
    _LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIRECTORY / f"{datetime.now():%Y%m%d}_{suffix}.json"

    records = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as log_file:
            records = json.load(log_file)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON array in {log_path}")

    records.append(record)
    with log_path.open("w", encoding="utf-8") as log_file:
        json.dump(records, log_file, ensure_ascii=False, indent=2)
        log_file.write("\n")
