import json

import pytest

from scripts.verify_coverage import (
    CoverageVerificationError,
    discover_production_files,
    format_report,
    normalize_coverage_path,
    verify_coverage,
)


def test_normalize_coverage_path():
    assert normalize_coverage_path("app/models/veo.py") == "app/models/veo.py"
    assert normalize_coverage_path(".\\app\\models\\veo.py") == "app/models/veo.py"
    assert normalize_coverage_path("C:/projects/videogen/app/models/veo.py") == "app/models/veo.py"
    assert (
        normalize_coverage_path("/home/runner/work/videogen/app/storage/google_drive.py")
        == "app/storage/google_drive.py"
    )
    assert (
        normalize_coverage_path("E:\\ML Projects\\videogen\\app\\video\\service.py")
        == "app/video/service.py"
    )


def test_discover_production_files(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "mod_a.py").write_text("# mod a")
    sub = app_dir / "sub"
    sub.mkdir()
    (sub / "mod_b.py").write_text("# mod b")
    (sub / "ignored.txt").write_text("not python")

    files = discover_production_files(app_dir)
    assert files == ["app/mod_a.py", "app/sub/mod_b.py"]


def test_verify_coverage_missing_artifact_raises(tmp_path):
    missing = tmp_path / "nonexistent.json"
    with pytest.raises(CoverageVerificationError, match="Coverage artifact not found"):
        verify_coverage(missing, tmp_path / "app", tmp_path)


def test_verify_coverage_malformed_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{broken json")
    with pytest.raises(CoverageVerificationError, match="malformed JSON"):
        verify_coverage(bad, tmp_path / "app", tmp_path)


def test_verify_coverage_missing_totals_or_files_raises(tmp_path):
    f1 = tmp_path / "no_totals.json"
    f1.write_text(json.dumps({"files": {}}))
    with pytest.raises(CoverageVerificationError, match="missing 'totals'"):
        verify_coverage(f1, tmp_path / "app", tmp_path)

    f2 = tmp_path / "no_files.json"
    f2.write_text(json.dumps({"totals": {}}))
    with pytest.raises(CoverageVerificationError, match="missing 'files'"):
        verify_coverage(f2, tmp_path / "app", tmp_path)


def test_verify_coverage_all_compliant(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "mod_a.py").write_text("")
    (app_dir / "mod_b.py").write_text("")

    cov_file = tmp_path / "coverage.json"
    payload = {
        "totals": {
            "num_statements": 100,
            "covered_lines": 95,
            "num_branches": 10,
            "covered_branches": 9,
        },
        "files": {
            "app/mod_a.py": {
                "summary": {
                    "num_statements": 50,
                    "covered_lines": 48,
                    "num_branches": 4,
                    "covered_branches": 4,
                }
            },
            "app/mod_b.py": {
                "summary": {
                    "num_statements": 50,
                    "covered_lines": 47,
                    "num_branches": 6,
                    "covered_branches": 5,
                }
            },
        },
    }
    cov_file.write_text(json.dumps(payload))

    passed, report = verify_coverage(cov_file, app_dir, tmp_path, threshold=0.90)
    assert passed is True
    assert report["overall"]["passed"] is True
    assert all(m["status"] == "PASSED" for m in report["modules"])


def test_verify_coverage_count_based_threshold_precision(tmp_path):
    """Verify that 89/99 (89.89%) strictly FAILS and 9/10 (90.0%) PASSES."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "failing_mod.py").write_text("")

    cov_file = tmp_path / "coverage.json"
    # 89 / 99 = 0.89898989...
    payload = {
        "totals": {"num_statements": 99, "covered_lines": 89},
        "files": {
            "app/failing_mod.py": {"summary": {"num_statements": 99, "covered_lines": 89}},
        },
    }
    cov_file.write_text(json.dumps(payload))

    passed, report = verify_coverage(cov_file, app_dir, tmp_path, threshold=0.90)
    assert passed is False
    assert report["overall"]["passed"] is False
    assert report["modules"][0]["status"] == "FAILED"
    assert report["modules"][0]["ratio"] == pytest.approx(89 / 99)

    # Exact boundary 9 / 10 = 0.90
    payload["totals"] = {"num_statements": 10, "covered_lines": 9}
    payload["files"]["app/failing_mod.py"]["summary"] = {"num_statements": 10, "covered_lines": 9}
    cov_file.write_text(json.dumps(payload))

    passed, report = verify_coverage(cov_file, app_dir, tmp_path, threshold=0.90)
    assert passed is True
    assert report["modules"][0]["status"] == "PASSED"
    assert report["modules"][0]["ratio"] == 0.90


def test_verify_coverage_single_module_below_threshold_fails_gate(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "good_mod.py").write_text("")
    (app_dir / "bad_mod.py").write_text("")

    cov_file = tmp_path / "coverage.json"
    # Overall is 90 / 100 = 90% (passes), but bad_mod is 30/40 = 75% (fails)
    payload = {
        "totals": {"num_statements": 100, "covered_lines": 90},
        "files": {
            "app/good_mod.py": {"summary": {"num_statements": 60, "covered_lines": 60}},
            "app/bad_mod.py": {"summary": {"num_statements": 40, "covered_lines": 30}},
        },
    }
    cov_file.write_text(json.dumps(payload))

    passed, report = verify_coverage(cov_file, app_dir, tmp_path, threshold=0.90)
    assert passed is False
    assert report["overall"]["passed"] is True
    statuses = {m["module"]: m["status"] for m in report["modules"]}
    assert statuses["app/good_mod.py"] == "PASSED"
    assert statuses["app/bad_mod.py"] == "FAILED"


def test_verify_coverage_deferred_module_does_not_fail(tmp_path):
    from unittest.mock import patch

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "good.py").write_text("")
    mpt_dir = app_dir / "mpt"
    mpt_dir.mkdir()
    (mpt_dir / "adapter.py").write_text("")

    cov_file = tmp_path / "coverage.json"
    payload = {
        "totals": {"num_statements": 10, "covered_lines": 10},
        "files": {
            "app/good.py": {"summary": {"num_statements": 10, "covered_lines": 10}},
            "app/mpt/adapter.py": {"summary": {"num_statements": 2, "covered_lines": 0}},
        },
    }
    cov_file.write_text(json.dumps(payload))

    with patch.dict(
        "scripts.verify_coverage.DEFERRED_MODULES",
        {"app/mpt/adapter.py": "Test deferred module"},
    ):
        passed, report = verify_coverage(cov_file, app_dir, tmp_path, threshold=0.90)
        assert passed is True
        mpt_res = next(m for m in report["modules"] if m["module"] == "app/mpt/adapter.py")
        assert mpt_res["status"] == "DEFERRED"


def test_verify_coverage_missing_file_fails_gate(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "tracked_mod.py").write_text("")
    (app_dir / "untested_mod.py").write_text("")

    cov_file = tmp_path / "coverage.json"
    # untested_mod.py is completely absent from coverage.json
    payload = {
        "totals": {"num_statements": 10, "covered_lines": 10},
        "files": {
            "app/tracked_mod.py": {"summary": {"num_statements": 10, "covered_lines": 10}},
        },
    }
    cov_file.write_text(json.dumps(payload))

    passed, report = verify_coverage(cov_file, app_dir, tmp_path, threshold=0.90)
    assert passed is False
    statuses = {m["module"]: m["status"] for m in report["modules"]}
    assert statuses["app/tracked_mod.py"] == "PASSED"
    assert statuses["app/untested_mod.py"] == "MISSING_DATA"

    formatted = format_report(report)
    assert "MISSING DATA" in formatted or "MISSING_DATA" in formatted
