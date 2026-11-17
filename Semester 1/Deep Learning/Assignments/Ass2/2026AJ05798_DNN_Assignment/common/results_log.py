"""Append experiment rows to results.csv."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from common.config import RESULTS_COLUMNS

_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = _ROOT / "results.csv"


def ensure_results_file() -> None:
    if not RESULTS_PATH.exists():
        with RESULTS_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULTS_COLUMNS).writeheader()


def log_result(row: dict[str, Any]) -> None:
    ensure_results_file()
    out = {k: row.get(k, "") for k in RESULTS_COLUMNS}
    with RESULTS_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULTS_COLUMNS).writerow(out)