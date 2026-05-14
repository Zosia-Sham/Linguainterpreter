"""Unit tests for src/lead_tools.py — read-only Lead Agent tools.

Exercises:
- Path traversal protection (``..``, absolute paths escaping project_root).
- Existing vs missing targets.
- Each tool's happy path against a temporary project_root.
- Arg-line parser (JSON, key=value, positional).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lead_tools import (  # noqa: E402
    LeadToolError,
    call_tool,
    list_tool_names,
    parse_args_line,
    tool_csv_head,
    tool_exists,
    tool_find,
    tool_grep,
    tool_ls,
    tool_parquet_schema,
    tool_read_text,
    tool_stat,
)


class _FakeOrch:
    def __init__(self, root: Path) -> None:
        self.project_root = root


@pytest.fixture
def fake_project() -> _FakeOrch:
    tmp = Path(tempfile.mkdtemp(prefix="lead_tools_test_"))
    (tmp / "artifacts").mkdir()
    (tmp / "artifacts" / "features.parquet").write_bytes(b"PAR1dummy")  # placeholder
    (tmp / "artifacts" / "notes.md").write_text("hello world\nsecond line\nneedle here\n", encoding="utf-8")
    (tmp / "data").mkdir()
    (tmp / "data" / "train.csv").write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    yield _FakeOrch(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Registry / arg parser
# ---------------------------------------------------------------------------
def test_registry_has_all_tools() -> None:
    names = set(list_tool_names())
    assert names == {
        "ls",
        "exists",
        "stat",
        "find",
        "parquet_schema",
        "csv_head",
        "read_text",
        "grep",
    }


def test_parse_args_json() -> None:
    assert parse_args_line('{"path": "artifacts/x.parquet"}') == {"path": "artifacts/x.parquet"}


def test_parse_args_kv() -> None:
    assert parse_args_line("path=artifacts/x.parquet n=10") == {
        "path": "artifacts/x.parquet",
        "n": "10",
    }


def test_parse_args_positional() -> None:
    assert parse_args_line("artifacts/x.parquet") == {"path": "artifacts/x.parquet"}


def test_parse_args_empty() -> None:
    assert parse_args_line("") == {}
    assert parse_args_line(None) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def test_traversal_blocked_via_dotdot(fake_project: _FakeOrch) -> None:
    with pytest.raises(LeadToolError, match="traversal"):
        tool_exists(fake_project, "../escape.txt")


def test_absolute_path_outside_root_blocked(fake_project: _FakeOrch) -> None:
    # An absolute path outside project_root must be rejected.
    outside = Path(tempfile.gettempdir()) / "definitely_outside_root_marker.txt"
    outside.write_text("x", encoding="utf-8")
    try:
        with pytest.raises(LeadToolError, match="escapes project_root"):
            tool_exists(fake_project, str(outside))
    finally:
        outside.unlink(missing_ok=True)


def test_empty_path_rejected(fake_project: _FakeOrch) -> None:
    with pytest.raises(LeadToolError, match="empty"):
        tool_exists(fake_project, "")


# ---------------------------------------------------------------------------
# Tools: happy paths
# ---------------------------------------------------------------------------
def test_exists_yes_no(fake_project: _FakeOrch) -> None:
    assert tool_exists(fake_project, "artifacts/features.parquet") == "true"
    assert tool_exists(fake_project, "artifacts/missing.parquet") == "false"


def test_stat_file(fake_project: _FakeOrch) -> None:
    out = tool_stat(fake_project, "artifacts/notes.md")
    assert "is_file: True" in out
    assert "is_dir: False" in out
    assert "size_bytes:" in out


def test_stat_missing_raises(fake_project: _FakeOrch) -> None:
    with pytest.raises(LeadToolError, match="not found"):
        tool_stat(fake_project, "nope/nada.md")


def test_ls_directory(fake_project: _FakeOrch) -> None:
    out = tool_ls(fake_project, "artifacts")
    assert "features.parquet" in out
    assert "notes.md" in out


def test_ls_on_file_raises(fake_project: _FakeOrch) -> None:
    with pytest.raises(LeadToolError, match="not a directory"):
        tool_ls(fake_project, "artifacts/notes.md")


def test_find_glob(fake_project: _FakeOrch) -> None:
    out = tool_find(fake_project, "*.parquet")
    assert "features.parquet" in out


def test_find_no_matches(fake_project: _FakeOrch) -> None:
    out = tool_find(fake_project, "*.nomatch")
    assert "no matches" in out


def test_read_text_head(fake_project: _FakeOrch) -> None:
    out = tool_read_text(fake_project, "artifacts/notes.md", n_bytes=500)
    assert "hello world" in out
    assert "needle here" in out


def test_grep_hit(fake_project: _FakeOrch) -> None:
    out = tool_grep(fake_project, "needle", "artifacts/notes.md")
    assert "needle here" in out
    assert ":" in out  # line number prefix


def test_grep_miss(fake_project: _FakeOrch) -> None:
    out = tool_grep(fake_project, "absent_token_xyz", "artifacts/notes.md")
    assert "no matches" in out


def test_csv_head_polars(fake_project: _FakeOrch) -> None:
    pytest.importorskip("polars")
    out = tool_csv_head(fake_project, "data/train.csv", n=2)
    assert "a" in out and "b" in out and "c" in out


def test_parquet_schema_polars_real_file(fake_project: _FakeOrch) -> None:
    # Build a real parquet file with polars so scan_parquet works.
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    target = Path(fake_project.project_root) / "artifacts" / "real.parquet"
    df.write_parquet(target)
    out = tool_parquet_schema(fake_project, "artifacts/real.parquet")
    assert "rows: 3" in out
    assert "columns: 2" in out
    assert "x" in out and "y" in out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def test_call_tool_dispatch(fake_project: _FakeOrch) -> None:
    assert call_tool(fake_project, "exists", {"path": "artifacts/notes.md"}) == "true"


def test_call_tool_unknown(fake_project: _FakeOrch) -> None:
    with pytest.raises(LeadToolError, match="unknown tool"):
        call_tool(fake_project, "rm -rf /", {"path": "/"})


def test_call_tool_bad_args(fake_project: _FakeOrch) -> None:
    with pytest.raises(LeadToolError, match="bad arguments"):
        call_tool(fake_project, "exists", {"not_a_real_param": "x"})


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-x", "-q"]))
