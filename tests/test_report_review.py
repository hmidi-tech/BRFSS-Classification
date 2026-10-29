"""Report consistency checks independent of private candidate artifacts."""

import json
from pathlib import Path
import shutil

import pytest

from build_report import generate, validate_measurements


def test_report_generation_uses_tracked_aggregates_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    results = tmp_path / "results"
    shutil.copytree(root / "results", results)
    config = json.loads((root / "config/selected_model.json").read_text())
    assert not (tmp_path / config["candidate_artifact"]).exists()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    text = generate(results, path, root / "artifacts/final_model.json")
    assert "\\newcommand{\\HoldFone}{0.4225}" in text
    assert "\\newcommand{\\FinalDimension}{1,183}" in text
    assert not (tmp_path / "latex-example-paper").exists()


def test_report_rejects_inconsistent_saved_metrics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "results", tmp_path / "results")
    path = tmp_path / "results/cross_validation.json"
    data = json.loads(path.read_text())
    data["folds"][0]["metrics"]["f1"] += 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="CV mean"):
        validate_measurements(tmp_path / "results", root / "config/selected_model.json")
