"""Unit tests for the BashAgent timeout-policy math.

Covers the regression that caused empty submissions in run
2026-05-08T02-10-14: the previous formula
``min(cascade_cap, max(floor, pred_timeout))`` silently truncated heavy ML
steps whose predictor estimate (e.g. 2880s) far exceeded the per-task budget
(e.g. 280s), killing the subprocess in ~280s and exhausting the run-wide
budget on retries before the final ``write_submission`` step could run.

The fix in ``BashAgent.run`` makes the predictor authoritative within the
hard-cap (run-wide remaining budget) instead of being truncated by the
per-task cascade budget. Formula:

    target    = max(cascade_cap, pred_timeout)
    effective = max(1, min(hard_cap, max(floor, target)))
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bash_agent import BashAgent  # noqa: E402


def _run_and_capture(
    *,
    timeout: int,
    prediction: dict | None,
    hard_cap: int | None,
    floor: int = 60,
    buffer_pct: int = 30,
):
    """Invoke BashAgent.run with subprocess.run patched to a no-op so we can
    inspect ``effective_timeout`` from the captured print output."""
    agent = BashAgent(
        workdir=str(ROOT),
        min_exec_timeout_sec=floor,
        predictive_buffer_pct=buffer_pct,
    )
    captured: list[str] = []

    def _fake_print(*args, **kwargs):  # mimic builtins.print signature
        captured.append(" ".join(str(a) for a in args))

    class _FakeCompletedProcess:
        stdout = ""
        stderr = ""
        returncode = 0

    with patch("src.bash_agent.subprocess.run", return_value=_FakeCompletedProcess()), \
         patch("builtins.print", side_effect=_fake_print):
        agent.run(
            "echo hi",
            timeout=timeout,
            prediction=prediction,
            hard_cap=hard_cap,
            stream=False,
        )

    # Find the [MONITOR] Predictive target line and parse effective_timeout.
    for line in captured:
        if "[MONITOR] Predictive target" in line:
            for tok in line.replace("→", " ").split():
                if tok.startswith("effective_timeout="):
                    return int(tok.split("=", 1)[1].rstrip("s"))
    pytest.fail(f"No [MONITOR] line captured. Output: {captured}")


def test_predictor_expands_beyond_cascade_cap():
    """Heavy ML step: predictor says 2400s + 30% buffer = 3120s; cascade_cap
    is the per-task budget (280s); hard_cap is run-wide remaining (7200s).
    The old formula returned 280; the new formula must honour the predictor.
    """
    eff = _run_and_capture(
        timeout=280,
        prediction={"expected_time_sec": 2400},
        hard_cap=7200,
        floor=60,
        buffer_pct=30,
    )
    # 2400 * 1.30 = 3120; well under the 7200s hard_cap, well above the 280s
    # cascade_cap and 60s floor — must equal 3120.
    assert eff == 3120, f"expected 3120s, got {eff}s"


def test_cascade_cap_wins_when_predictor_estimates_less():
    """Light step: predictor says 60s + 30% = 78s, cascade_cap=280s. The
    soft target = max(280, 78) = 280, under hard_cap, above floor → 280s.
    """
    eff = _run_and_capture(
        timeout=280,
        prediction={"expected_time_sec": 60},
        hard_cap=7200,
        floor=60,
        buffer_pct=30,
    )
    assert eff == 280, f"expected 280s, got {eff}s"


def test_hard_cap_clamps_predictor():
    """Run-wide budget almost exhausted: hard_cap=200s wins over both
    cascade_cap and predictor — we must never exceed the absolute remaining
    budget.
    """
    eff = _run_and_capture(
        timeout=280,
        prediction={"expected_time_sec": 2400},
        hard_cap=200,
        floor=60,
        buffer_pct=30,
    )
    assert eff == 200, f"expected 200s (hard_cap), got {eff}s"


def test_no_prediction_uses_cascade_cap_floor():
    """No predictor available → effective = max(floor, cascade_cap) clamped
    by hard_cap. With cascade_cap=900 and floor=60, hard_cap=7200 → 900.
    """
    eff = _run_and_capture(
        timeout=900,
        prediction=None,
        hard_cap=7200,
        floor=60,
        buffer_pct=30,
    )
    assert eff == 900, f"expected 900s, got {eff}s"


def test_no_cascade_cap_uses_predictor():
    """No cascade cap, only prediction → buffered prediction wins (within
    hard_cap and floor)."""
    eff = _run_and_capture(
        timeout=None,  # type: ignore[arg-type]
        prediction={"expected_time_sec": 1000},
        hard_cap=7200,
        floor=60,
        buffer_pct=30,
    )
    # 1000 * 1.30 = 1300
    assert eff == 1300, f"expected 1300s, got {eff}s"


def test_floor_applies_when_everything_smaller():
    """Tiny prediction and tiny cascade_cap → floor wins."""
    eff = _run_and_capture(
        timeout=10,
        prediction={"expected_time_sec": 5},
        hard_cap=7200,
        floor=60,
        buffer_pct=30,
    )
    assert eff == 60, f"expected 60s (floor), got {eff}s"


def test_hard_cap_overrides_floor_when_smaller():
    """Edge case: run-wide remaining < floor. Hard cap is absolute and must
    win even over the floor (we never exceed remaining global budget)."""
    eff = _run_and_capture(
        timeout=280,
        prediction={"expected_time_sec": 2400},
        hard_cap=30,
        floor=60,
        buffer_pct=30,
    )
    assert eff == 30, f"expected 30s (hard_cap < floor), got {eff}s"
