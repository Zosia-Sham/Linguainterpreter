"""Unit tests for _verify_claimed_artifacts in src/pipeline.py.

Covers the hallucination-guard flow: extracting claimed save paths from
stdout, classifying them as verified (exist on disk) or missing.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import _verify_claimed_artifacts  # noqa: E402


class _FakeOrch:
    def __init__(self, root: Path) -> None:
        self.project_root = root


@pytest.fixture
def orch() -> _FakeOrch:
    tmp = Path(tempfile.mkdtemp(prefix="verify_test_"))
    (tmp / "artifacts").mkdir()
    (tmp / "artifacts" / "good.parquet").write_bytes(b"data")
    (tmp / "artifacts" / "metrics.json").write_text("{}", encoding="utf-8")
    yield _FakeOrch(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


def test_saved_to_verbatim(orch: _FakeOrch) -> None:
    blob = "Saved to artifacts/good.parquet"
    res = _verify_claimed_artifacts(orch, [blob])
    assert "artifacts/good.parquet" in res["verified"]
    assert res["missing"] == []


def test_saved_with_words_between(orch: _FakeOrch) -> None:
    """Real hallucination pattern from basketball-2026-exp5-pafnvc.log."""
    blob = "Saved the engineered features to artifacts/fe_level3_target_encoded.parquet"
    res = _verify_claimed_artifacts(orch, [blob])
    assert res["verified"] == []
    assert "artifacts/fe_level3_target_encoded.parquet" in res["missing"]


def test_mixed_verified_and_missing(orch: _FakeOrch) -> None:
    blob = (
        "Wrote artifacts/good.parquet\n"
        "-> artifacts/does_not_exist.csv\n"
        "Output: artifacts/metrics.json\n"
    )
    res = _verify_claimed_artifacts(orch, [blob])
    assert set(res["verified"]) == {"artifacts/good.parquet", "artifacts/metrics.json"}
    assert "artifacts/does_not_exist.csv" in res["missing"]


def test_dumped_to_pattern(orch: _FakeOrch) -> None:
    blob = "Exported metrics to artifacts/metrics.json"
    res = _verify_claimed_artifacts(orch, [blob])
    assert "artifacts/metrics.json" in res["verified"]


def test_empty_blobs(orch: _FakeOrch) -> None:
    res = _verify_claimed_artifacts(orch, ["", None])  # type: ignore[list-item]
    assert res == {"verified": [], "missing": []}


def test_path_outside_project_root_ignored(orch: _FakeOrch) -> None:
    blob = "Saved to /etc/passwd.txt"
    res = _verify_claimed_artifacts(orch, [blob])
    # /etc/passwd.txt is outside project_root — must NOT show up in verified.
    assert "/etc/passwd.txt" not in res["verified"]


def test_url_skipped(orch: _FakeOrch) -> None:
    blob = "Uploaded to https://example.com/bucket/file.parquet"
    res = _verify_claimed_artifacts(orch, [blob])
    assert res["verified"] == []
    assert res["missing"] == []


def test_deduplicates(orch: _FakeOrch) -> None:
    blob = "Saved to artifacts/good.parquet\nWrote artifacts/good.parquet"
    res = _verify_claimed_artifacts(orch, [blob])
    assert res["verified"].count("artifacts/good.parquet") == 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-x", "-q"]))
