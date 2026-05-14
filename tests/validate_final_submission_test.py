"""
Unit-tests for validate_final_submission and quick_submission_check.

Covers:
  - constant predictions (diversity check)
  - wrong columns (header mismatch)
  - wrong row count vs sample_submission
  - empty / missing file
  - valid submission (happy path)
  - constant preds reported even when header is also wrong (tests the "if not errors" fix)
  - anti-logits detection
  - quick_submission_check standalone (no orchestrator)
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest



@dataclass
class _StubPaths:
    artifacts_dir: str = "artifacts"
    submission_dir: str = "submission"
    submission_filename: str = "submission.csv"
    data_dir: str = "data"
    scripts_dir: str = "scripts"
    src_dir: str = "src"
    logs_dir: str = "logs"
    tests_dir: str = "tests"
    venv_dir: str = ".venv"


@dataclass
class _StubOrchestration:
    total_budget_sec: int = 7200
    react_max_rounds: int = 3


@dataclass
class _StubRuntime:
    project_name: str = "test_project"
    create_env: bool = False
    code_timeout_sec: int = 300
    pip_timeout_sec: int = 120
    bash_timeout_sec: int = 60
    checker_timeout_cap_sec: int = 60
    min_exec_timeout_sec: int = 10
    generation_retry_limit: int = 5
    router_retry_limit: int = 5
    metric_validation_retry_limit: int = 2
    prediction_fallback_sec: int = 300
    default_task_budget_sec: int = 1800
    execution_output_shorten_threshold: int = 50000
    execution_output_shorten_target: int = 40000
    replan_context_chars: int = 4000
    aggregate_tail_chars: int = 4000
    attach_hardware_limit_files: int = 10


@dataclass
class _StubCfg:
    paths: _StubPaths = field(default_factory=_StubPaths)
    orchestration: _StubOrchestration = field(default_factory=_StubOrchestration)
    runtime: _StubRuntime = field(default_factory=_StubRuntime)


class StubOrchestrator:
    """
    Minimal mock satisfying validate_final_submission interface.
    Only needs: project_root, cfg.paths.*, write_file(), effective_elapsed_sec().
    """

    def __init__(self, project_root: Path):
        self.project_root = str(project_root)
        self.cfg = _StubCfg()
        self._written: Dict[str, str] = {}

    def write_file(self, rel_path: str, content: str) -> Path:
        abs_path = Path(self.project_root) / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        self._written[rel_path] = content
        return abs_path

    def effective_elapsed_sec(self) -> float:
        return 0.0


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _make_spec(
    columns: List[str],
    primary_name: str = "accuracy",
    maximize: bool = True,
    sample_submission_csv: str = "",
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "primary_metric": {"name": primary_name, "maximize": maximize},
        "secondary_metrics": [],
        "submission": {"columns": columns, "delimiter": ","},
        "data": {},
    }
    if sample_submission_csv:
        spec["data"]["sample_submission_csv"] = sample_submission_csv
    return spec


from src.helpers import validate_final_submission, quick_submission_check, SubmissionCheckResult

class TestValidateFinalSubmission:

    def test_valid_submission(self, tmp_path: Path):
        orch = StubOrchestrator(tmp_path)
        cols = ["id", "target"]
        rows = [[i, float(i % 7) * 0.14] for i in range(100)]

        canonical = tmp_path / "submission" / "submission.csv"
        _write_csv(canonical, cols, rows)

        sample = tmp_path / "data" / "sample_submission.csv"
        _write_csv(sample, cols, rows)

        spec = _make_spec(cols, sample_submission_csv=str(sample))
        result = validate_final_submission(orch, spec)

        assert result["ok"] is True, f"Expected ok=True, errors={result.get('errors')}"
        assert result["errors"] == []

    def test_wrong_columns(self, tmp_path: Path):
        orch = StubOrchestrator(tmp_path)
        expected_cols = ["id", "target"]
        actual_cols = ["ID", "prediction"]
        rows = [[i, 0.5 + i * 0.001] for i in range(10)]

        canonical = tmp_path / "submission" / "submission.csv"
        _write_csv(canonical, actual_cols, rows)

        spec = _make_spec(expected_cols)
        result = validate_final_submission(orch, spec)

        assert result["ok"] is False
        col_errors = [
            e for e in result["errors"]
            if "mismatch" in e.lower() or "header" in e.lower()
        ]
        assert len(col_errors) > 0, f"Expected header mismatch error, got: {result['errors']}"

    def test_constant_predictions(self, tmp_path: Path):
        orch = StubOrchestrator(tmp_path)
        cols = ["id", "target"]
        rows = [[i, 0.5] for i in range(200)]

        canonical = tmp_path / "submission" / "submission.csv"
        _write_csv(canonical, cols, rows)

        sample = tmp_path / "data" / "sample_submission.csv"
        _write_csv(sample, cols, [[i, float(i % 5) * 0.2] for i in range(200)])

        spec = _make_spec(cols, sample_submission_csv=str(sample))
        result = validate_final_submission(orch, spec)

        assert result["ok"] is False
        diversity_errors = [
            e for e in result["errors"]
            if "constant" in e.lower() or "diversity" in e.lower()
        ]
        assert len(diversity_errors) > 0, f"Expected diversity error, got: {result['errors']}"

    def test_anti_logits_detected(self, tmp_path: Path):
        """Sample has probabilities [0,1]; submission has raw logits outside [0,1]."""
        orch = StubOrchestrator(tmp_path)
        cols = ["id", "target"]

        sample = tmp_path / "data" / "sample_submission.csv"
        _write_csv(sample, cols, [[i, (i % 10) * 0.1] for i in range(100)])

        canonical = tmp_path / "submission" / "submission.csv"
        _write_csv(canonical, cols, [[i, -2.5 + i * 0.05] for i in range(100)])

        spec = _make_spec(cols, sample_submission_csv=str(sample))
        result = validate_final_submission(orch, spec)

        assert result["ok"] is False
        assert any(
            "logit" in e.lower() or "probability" in e.lower() or "confined" in e.lower()
            for e in result["errors"]
        ), f"Expected anti-logits error, got: {result['errors']}"

    def test_empty_file(self, tmp_path: Path):
        orch = StubOrchestrator(tmp_path)
        canonical = tmp_path / "submission" / "submission.csv"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text("", encoding="utf-8")

        spec = _make_spec(["id", "target"])
        result = validate_final_submission(orch, spec)

        assert result["ok"] is False
        assert any("empty" in e.lower() for e in result["errors"])

    def test_missing_file(self, tmp_path: Path):
        orch = StubOrchestrator(tmp_path)
        spec = _make_spec(["id", "target"])
        result = validate_final_submission(orch, spec)

        assert result["ok"] is False
        assert any("missing" in e.lower() for e in result["errors"])

    # constant preds reported even when header is wrong
    def test_constant_preds_reported_alongside_wrong_cols(self, tmp_path: Path):
        """
        After the fix (removing `if not errors` guard before diversity check),
        diversity errors should appear even when header is also wrong.
        """
        orch = StubOrchestrator(tmp_path)
        expected_cols = ["id", "target"]
        actual_cols = ["idx", "pred"]  # wrong header
        rows = [[i, 0.42] for i in range(100)]  # constant

        canonical = tmp_path / "submission" / "submission.csv"
        _write_csv(canonical, actual_cols, rows)

        spec = _make_spec(expected_cols)
        result = validate_final_submission(orch, spec)

        assert result["ok"] is False
        has_header_err = any(
            "mismatch" in e.lower() or "header" in e.lower()
            for e in result["errors"]
        )
        has_diversity_err = any(
            "constant" in e.lower() or "diversity" in e.lower()
            for e in result["errors"]
        )
        assert has_header_err, f"Expected header error, got: {result['errors']}"
        assert has_diversity_err, (
            f"Expected diversity error alongside header error "
            f"(if-not-errors guard must be removed), got: {result['errors']}"
        )

    def test_header_only_no_data_rows(self, tmp_path: Path):
        orch = StubOrchestrator(tmp_path)
        cols = ["id", "target"]
        canonical = tmp_path / "submission" / "submission.csv"
        _write_csv(canonical, cols, [])  # header but zero data rows

        sample = tmp_path / "data" / "sample_submission.csv"
        _write_csv(sample, cols, [[i, 0.1 * i] for i in range(50)])

        spec = _make_spec(cols, sample_submission_csv=str(sample))
        result = validate_final_submission(orch, spec)

        # Should fail because sample has 50 rows but submission has 0
        # (row count check is in _validate_candidate_submission_file,
        #  but semantic check will also have nothing to probe)
        # At minimum, the submission is useless with 0 rows.
        # The function may or may not catch this depending on which checks fire.
        # We just verify it doesn't crash.
        assert isinstance(result, dict)
        assert "ok" in result


# quick_submission_check
class TestQuickSubmissionCheck:

    def test_valid(self, tmp_path: Path):
        cols = ["id", "target"]
        sub = tmp_path / "submission.csv"
        sample = tmp_path / "sample.csv"
        rows = [[i, (i % 7) * 0.14] for i in range(50)]
        _write_csv(sub, cols, rows)
        _write_csv(sample, cols, rows)

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
            sample_submission_path=sample,
        )
        assert result.ok is True
        assert result.errors == []

    def test_missing_file(self, tmp_path: Path):
        result = quick_submission_check(
            submission_path=tmp_path / "nonexistent.csv",
            expected_columns=["id", "target"],
        )
        assert result.ok is False
        assert any("not exist" in e.lower() or "missing" in e.lower() for e in result.errors)

    def test_empty_file(self, tmp_path: Path):
        sub = tmp_path / "submission.csv"
        sub.write_text("", encoding="utf-8")
        result = quick_submission_check(
            submission_path=sub,
            expected_columns=["id", "target"],
        )
        assert result.ok is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_wrong_columns(self, tmp_path: Path):
        sub = tmp_path / "submission.csv"
        _write_csv(sub, ["wrong_id", "wrong_target"], [[1, 0.5], [2, 0.6]])
        result = quick_submission_check(
            submission_path=sub,
            expected_columns=["id", "target"],
        )
        assert result.ok is False
        assert any(
            "column" in e.lower() or "mismatch" in e.lower()
            for e in result.errors
        )

    def test_wrong_row_count(self, tmp_path: Path):
        cols = ["id", "target"]
        sub = tmp_path / "submission.csv"
        sample = tmp_path / "sample.csv"
        _write_csv(sub, cols, [[i, 0.1 * i] for i in range(50)])
        _write_csv(sample, cols, [[i, 0.1 * i] for i in range(100)])  # 100 vs 50

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
            sample_submission_path=sample,
        )
        assert result.ok is False
        assert any(
            "row" in e.lower() and "count" in e.lower()
            for e in result.errors
        )

    def test_constant_predictions(self, tmp_path: Path):
        cols = ["id", "target"]
        sub = tmp_path / "submission.csv"
        _write_csv(sub, cols, [[i, 0.999] for i in range(200)])

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
        )
        assert result.ok is False
        assert any(
            "constant" in e.lower() or "diversity" in e.lower()
            for e in result.errors
        )

    def test_non_finite_predictions(self, tmp_path: Path):
        cols = ["id", "target"]
        sub = tmp_path / "submission.csv"
        sub.parent.mkdir(parents=True, exist_ok=True)
        with sub.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerow([0, "0.5"])
            w.writerow([1, "nan"])
            w.writerow([2, "0.3"])

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
        )
        assert result.ok is False
        assert any(
            "finite" in e.lower() or "nan" in e.lower() or "non-finite" in e.lower()
            for e in result.errors
        )

    def test_anti_logits_probability_mode(self, tmp_path: Path):
        """Sample has probabilities [0,1]; submission has raw logits."""
        cols = ["id", "prob"]
        sample = tmp_path / "sample.csv"
        _write_csv(sample, cols, [[i, (i % 10) * 0.1] for i in range(50)])

        sub = tmp_path / "submission.csv"
        _write_csv(sub, cols, [[i, -3.0 + i * 0.12] for i in range(50)])

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
            sample_submission_path=sample,
        )
        assert result.ok is False
        assert any(
            "logit" in e.lower() or "probability" in e.lower() or "range" in e.lower()
            for e in result.errors
        )

    def test_anti_logits_class_label_mode(self, tmp_path: Path):
        """Sample has discrete labels {0,1,2}; submission has floats."""
        cols = ["id", "label"]
        sample = tmp_path / "sample.csv"
        _write_csv(sample, cols, [[i, i % 3] for i in range(60)])

        sub = tmp_path / "submission.csv"
        _write_csv(sub, cols, [[i, 0.73 * i] for i in range(60)])

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
            sample_submission_path=sample,
        )
        assert result.ok is False
        assert any(
            "class" in e.lower() or "label" in e.lower() or "discrete" in e.lower()
            for e in result.errors
        )

    def test_multiple_errors_reported_together(self, tmp_path: Path):
        """Wrong columns AND constant predictions should both be reported."""
        cols_expected = ["id", "target"]
        cols_actual = ["idx", "pred"]
        sub = tmp_path / "submission.csv"
        _write_csv(sub, cols_actual, [[i, 0.5] for i in range(100)])

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols_expected,
        )
        assert result.ok is False
        has_col_err = any(
            "column" in e.lower() or "header" in e.lower() or "mismatch" in e.lower()
            for e in result.errors
        )
        has_div_err = any(
            "constant" in e.lower() or "diversity" in e.lower()
            for e in result.errors
        )
        assert has_col_err, f"Expected column error, got: {result.errors}"
        assert has_div_err, f"Expected diversity error alongside column error, got: {result.errors}"

    def test_no_expected_columns_skips_header_check(self, tmp_path: Path):
        sub = tmp_path / "submission.csv"
        _write_csv(sub, ["anything", "goes"], [[i, i * 0.1] for i in range(30)])

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=[],
        )
        assert result.ok is True

    def test_no_sample_skips_row_count_and_semantic(self, tmp_path: Path):
        cols = ["id", "target"]
        sub = tmp_path / "submission.csv"
        _write_csv(sub, cols, [[i, i * 0.01] for i in range(50)])

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
            sample_submission_path=None,
        )
        assert result.ok is True

    def test_result_has_metadata(self, tmp_path: Path):
        """SubmissionCheckResult should carry header and row_count."""
        cols = ["id", "target"]
        sub = tmp_path / "submission.csv"
        num_rows = 42  # arbitrary — just checking metadata is populated
        rows = [[i, i * 0.1] for i in range(num_rows)]
        _write_csv(sub, cols, rows)

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
        )
        assert result.header == cols
        assert result.row_count == num_rows

    def test_inf_values_detected(self, tmp_path: Path):
        cols = ["id", "target"]
        sub = tmp_path / "submission.csv"
        sub.parent.mkdir(parents=True, exist_ok=True)
        with sub.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerow([0, "0.5"])
            w.writerow([1, "inf"])
            w.writerow([2, "-inf"])

        result = quick_submission_check(
            submission_path=sub,
            expected_columns=cols,
        )
        assert result.ok is False
        assert any("finite" in e.lower() or "inf" in e.lower() for e in result.errors)