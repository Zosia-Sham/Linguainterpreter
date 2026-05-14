"""Smoke test for dict-concat LLM output recovery (non-YAML fallback)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import _try_parse_tasks_non_yaml  # noqa: E402

SAMPLE = r"""{'task': 'Load all 35 CSV files from C:\\Users\\neron\\Documents\\linguainterpreter\\data using Polars; document complete schema for each file including dtypes, null counts, and row counts; verify file relationships and primary keys (Season+DayNum+WTeamID+LTeamID for games, TeamID for teams); check data completeness by season with per-season row counts; detect duplicate games and impossible scores (negative scores, scores > 200, WScore < LScore); map team ID ranges confirming M: 1000-1999, W: 3000-3999; save artifacts: data_audit_report.json with completeness stats and quality flags, schema_documentation.md with full data dictionary.', 'time_budget_sec': 1800}{'task': 'Analyze Season/DayNum distributions across all game files; verify chronological ordering within each season; identify prediction-time data availability (DayNum <= 132 for pre-tournament, DayNum >= 134 for tournament games); check for future information leakage risks in potential features (ensure no use of post-prediction data); analyze tournament vs regular season timing patterns and DayNum alignment with Massey Ordinals RankingDayNum; document temporal constraints for feature engineering in temporal_analysis.md with leakage_check_report.json flagging any temporal violations found.', 'time_budget_sec': 1800}{'task': "Build training labels from historical tournament results (MNCAATourneyCompactResults, WNCAATourneyCompactResults) with binary outcome (1 if lower TeamID wins, 0 otherwise); analyze win rate distributions and check for class imbalance; perform adversarial validation: train LightGBM classifier to distinguish historical train seasons from 2026 prediction scenario using season-level aggregates, flag if AUC > 0.6 indicating distribution shift; analyze men's vs women's data differences in game counts, scoring patterns, and season lengths; save artifacts: target_analysis.json with label statistics and adversarial_val_report.md with shift analysis.", 'time_budget_sec': 1800}{'task': 'Implement strict time-series validation strategy: train on seasons < S, validate on season S for S in [2021, 2022, 2023, 2024, 2025]; test expanding window vs sliding window approaches; verify zero future data leakage in feature construction with automated checks; document CV-LB correlation plan for Stage 1 evaluation using 2022-2025 as public LB proxy; create reusable CV split generator in cv_strategy.py with validation_folds.json specifying exact train/val season ranges per fold.', 'time_budget_sec': 1800}"""


def test_split_concatenated_dict_reprs():
    out = _try_parse_tasks_non_yaml(SAMPLE)
    assert out is not None
    assert len(out) == 4
    assert all(isinstance(x, dict) and "task" in x for x in out)
    assert out[0]["time_budget_sec"] == 1800


if __name__ == "__main__":
    test_split_concatenated_dict_reprs()
    print("ok")
