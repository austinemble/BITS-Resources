"""Drop the short runs and repeat them with the same budget as the main experiments."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import RESULTS_COLUMNS


def clean_results() -> None:
    path = ROOT / "results.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    drop_notes = {
        "shakespeare",
        "heads=1",
        "heads=4",
        "heads=8",
        "no_causal_mask",
        "attn_map_run",
        "no_mask_sample",
    }
    seen: dict[tuple, int] = {}
    kept = []
    for r in rows:
        if r["notes"] in drop_notes:
            continue
        if r["model"].startswith("transformer_har") or "_h" in r["model"]:
            continue
        if r["notes"] == "adding" and r["model"] in {"rnn", "lstm", "gru"}:
            key = (r["model"], r["seed"], r["seq_len"])
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                continue
        kept.append(r)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        w.writerows(kept)
    print(f"kept {len(kept)} rows", flush=True)


def main() -> None:
    clean_results()
    from problem1.run_pipeline import run_hidden_size_sweep
    from problem2.run_pipeline import run_ablations, run_har_transformer, run_lm

    run_har_transformer()
    run_lm()
    run_ablations()
    run_hidden_size_sweep()
    print("FAIR RERUN DONE", flush=True)


if __name__ == "__main__":
    main()
