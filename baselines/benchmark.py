"""
benchmark.py — run multiple competitions sequentially using runner.BenchmarkRun.

Config file format (competitions.json):
{
    "competitions": [
        {"name": "comp_a"},
        {"name": "comp_b", "timeout": 30},
        {"name": "comp_c"}
    ],
    "default_timeout": 15
}

Usage:
    python benchmark.py                          # uses competitions.json
    python benchmark.py --config my_bench.json
    python benchmark.py --config my_bench.json --dry-run
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from runner import BenchmarkRun, DEFAULT_TIMEOUT_MINUTES


# ── config ─────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = Path(__file__).parent / "competitions.json"


def load_config(config_path: Path) -> tuple[list[dict], int]:
    """
    Returns (competitions, default_timeout_minutes).

    competitions is a list of dicts with at least {"name": str}
    and an optional {"timeout": int}.
    """
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)

    default_timeout = raw.get("default_timeout", DEFAULT_TIMEOUT_MINUTES)
    competitions = raw.get("competitions", [])

    if not competitions:
        raise ValueError(f"No competitions listed in {config_path}")

    return competitions, default_timeout


# ── runner ──────────────────────────────────────────────────────────────────────

def run_all(competitions: list[dict], default_timeout: int) -> list[dict]:
    """
    Run each competition sequentially.
    Returns a list of result dicts (one per competition).
    A failed/crashed run is recorded with status="error" and does NOT
    abort the rest of the benchmark.
    """
    results = []
    total = len(competitions)

    for i, comp in enumerate(competitions, start=1):
        name = comp["name"]
        timeout = comp.get("timeout", default_timeout)

        print(f"\n{'='*60}")
        print(f"  [{i}/{total}]  Competition: {name}  (timeout: {timeout} min)")
        print(f"{'='*60}\n")

        try:
            run = BenchmarkRun(competition_name=name, timeout_minutes=timeout)
            run.setup()
            run_info = run.run_claude()
            submission_found = run.check_submission()
            run.save_metadata(run_info, submission_found)
            run.print_summary(run_info, submission_found)

            results.append({
                "competition":      name,
                "status":           "ok",
                "runtime_sec":      run_info["runtime_sec"],
                "timed_out":        run_info["timed_out"],
                "submission_found": submission_found,
                "run_dir":          str(run.run_dir),
                "session_id":       run_info["session_id"],
            })

        except Exception as exc:
            print(f"\n[ERROR] Competition '{name}' crashed: {exc}")
            traceback.print_exc()
            results.append({
                "competition":      name,
                "status":           "error",
                "error":            str(exc),
                "runtime_sec":      None,
                "timed_out":        None,
                "submission_found": False,
                "run_dir":          None,
                "session_id":       None,
            })

    return results


# ── reporting ───────────────────────────────────────────────────────────────────

_STATUS_ICON = {
    True:  "✓",
    False: "✗",
    None:  "?",
}


def print_summary_table(results: list[dict]):
    col_w = [20, 8, 10, 10, 10]
    header = ["Competition", "Status", "Runtime", "TimedOut", "Submission"]
    sep = "  ".join("-" * w for w in col_w)

    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(sep)

    ok = timed = errors = submissions = 0

    for r in results:
        status_str = r["status"]
        runtime_str = f"{r['runtime_sec']}s" if r["runtime_sec"] is not None else "—"
        timeout_str = _STATUS_ICON.get(r["timed_out"], "?")
        sub_str = _STATUS_ICON[r["submission_found"]]

        row = [
            r["competition"][:col_w[0]],
            status_str[:col_w[1]],
            runtime_str[:col_w[2]],
            timeout_str,
            sub_str,
        ]
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))

        if r["status"] == "ok":
            ok += 1
        else:
            errors += 1
        if r.get("timed_out"):
            timed += 1
        if r["submission_found"]:
            submissions += 1

    total = len(results)
    print(f"\n  Total: {total}  |  OK: {ok}  |  Errors: {errors}  "
          f"|  Timed out: {timed}  |  Submissions: {submissions}/{total}")


def save_benchmark_results(results: list[dict], config_path: Path):
    out_path = config_path.parent / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "config":    str(config_path),
        "results":   results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {out_path}")


# ── entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sequential benchmark runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to competitions.json (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without running anything",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    competitions, default_timeout = load_config(args.config)

    print(f"Benchmark config: {args.config}")
    print(f"Default timeout:  {default_timeout} min")
    print(f"Competitions ({len(competitions)}):")
    for c in competitions:
        t = c.get("timeout", default_timeout)
        print(f"  • {c['name']}  (timeout: {t} min)")

    if args.dry_run:
        print("\n[dry-run] Exiting without running.")
        return

    results = run_all(competitions, default_timeout)
    print_summary_table(results)
    save_benchmark_results(results, args.config)


if __name__ == "__main__":
    main()
