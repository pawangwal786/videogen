#!/usr/bin/env python3
"""Coverage Quality Gate Verifier.

Enforces:
1. Overall statement coverage >= threshold (default 90.0%).
2. Every non-deferred production module in app/ >= threshold statement coverage.
3. Every discovered production module MUST be present in coverage.json (missing data = FAIL).
4. Strictly evaluates covered / total counts, not rounded percentage displays.
5. Path normalization handles absolute paths, Windows backslashes, and relative paths portably.
6. Reports branch coverage as diagnostic telemetry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFERRED_MODULES: dict[str, str] = {
    "app/mpt/adapter.py": "Deferred placeholder boundary; implemented & tested in Phase 6",
}


def normalize_coverage_path(raw_path: str, repo_root: Path | None = None) -> str:
    """Normalize a file path from coverage.json or filesystem to canonical 'app/...' POSIX path."""
    posix_path = raw_path.replace("\\", "/")

    # If it contains '/app/' or starts with 'app/'
    if "/app/" in posix_path:
        idx = posix_path.index("/app/")
        return posix_path[idx + 1 :]
    if posix_path.startswith("app/"):
        return posix_path

    # Try resolving relative to repo_root
    if repo_root is not None:
        try:
            resolved = Path(raw_path).resolve()
            root_resolved = repo_root.resolve()
            rel = resolved.relative_to(root_resolved).as_posix()
            if rel.startswith("app/"):
                return rel
        except (ValueError, OSError):
            return posix_path.strip("./")

    return posix_path.strip("./")


def discover_production_files(app_dir: Path) -> list[str]:
    """Discover all production Python files under app_dir."""
    if not app_dir.exists() or not app_dir.is_dir():
        return []
    files: list[str] = []
    for p in sorted(app_dir.rglob("*.py")):
        # Canonical relative path starting with 'app/'
        rel = p.relative_to(app_dir.parent).as_posix()
        files.append(rel)
    return files


class CoverageVerificationError(Exception):
    """Raised when coverage validation fails unexpectedly."""


def verify_coverage(
    coverage_file: Path,
    app_dir: Path,
    repo_root: Path,
    threshold: float = 0.90,
) -> tuple[bool, dict[str, Any]]:
    """Verify coverage JSON data against the quality gate.

    Returns:
        (passed, report_data)
    """
    if not coverage_file.exists():
        raise CoverageVerificationError(
            f"Coverage artifact not found at '{coverage_file}'. Run pytest with coverage JSON output first."
        )

    try:
        data = json.loads(coverage_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CoverageVerificationError(
            f"Coverage artifact '{coverage_file}' is malformed JSON: {e}"
        ) from e

    if not isinstance(data, dict):
        raise CoverageVerificationError(
            f"Coverage artifact '{coverage_file}' root is not a JSON object."
        )

    if "totals" not in data or not isinstance(data["totals"], dict):
        raise CoverageVerificationError("Coverage artifact missing 'totals' dictionary.")

    if "files" not in data or not isinstance(data["files"], dict):
        raise CoverageVerificationError("Coverage artifact missing 'files' dictionary.")

    # 1. Discover production files on disk
    prod_files = discover_production_files(app_dir)
    if not prod_files:
        raise CoverageVerificationError(f"No production Python files discovered in '{app_dir}'.")

    # 2. Build normalized map of coverage entries
    normalized_coverage: dict[str, dict[str, Any]] = {}
    for raw_path, file_data in data["files"].items():
        if not isinstance(file_data, dict) or "summary" not in file_data:
            continue
        norm = normalize_coverage_path(raw_path, repo_root)
        normalized_coverage[norm] = file_data

    # 3. Check overall coverage
    totals = data["totals"]
    total_stmts = totals.get("num_statements")
    total_covered = totals.get("covered_lines")

    if total_stmts is None or total_covered is None:
        raise CoverageVerificationError(
            "Totals dictionary missing 'num_statements' or 'covered_lines'."
        )

    overall_ratio = (total_covered / total_stmts) if total_stmts > 0 else 1.0
    overall_passed = overall_ratio >= threshold

    total_branches = totals.get("num_branches", 0)
    total_covered_branches = totals.get("covered_branches", 0)
    overall_branch_ratio = (total_covered_branches / total_branches) if total_branches > 0 else 1.0

    # 4. Check each production module
    module_results: list[dict[str, Any]] = []
    any_module_failed = False

    for prod_file in prod_files:
        if prod_file in DEFERRED_MODULES:
            module_results.append(
                {
                    "module": prod_file,
                    "status": "DEFERRED",
                    "reason": DEFERRED_MODULES[prod_file],
                    "statements": 0,
                    "covered": 0,
                    "ratio": 1.0,
                    "branches": 0,
                    "covered_branches": 0,
                    "branch_ratio": 1.0,
                }
            )
            continue

        if prod_file not in normalized_coverage:
            any_module_failed = True
            module_results.append(
                {
                    "module": prod_file,
                    "status": "MISSING_DATA",
                    "reason": "File exists on disk but absent in coverage.json",
                    "statements": 0,
                    "covered": 0,
                    "ratio": 0.0,
                    "branches": 0,
                    "covered_branches": 0,
                    "branch_ratio": 0.0,
                }
            )
            continue

        file_cov = normalized_coverage[prod_file]
        summary = file_cov.get("summary", {})
        stmts = summary.get("num_statements")
        covered = summary.get("covered_lines")

        if stmts is None or covered is None:
            any_module_failed = True
            module_results.append(
                {
                    "module": prod_file,
                    "status": "INVALID_METRICS",
                    "reason": "Summary missing num_statements or covered_lines",
                    "statements": 0,
                    "covered": 0,
                    "ratio": 0.0,
                    "branches": 0,
                    "covered_branches": 0,
                    "branch_ratio": 0.0,
                }
            )
            continue

        # Count-based ratio check
        ratio = (covered / stmts) if stmts > 0 else 1.0
        passed = ratio >= threshold

        if not passed:
            any_module_failed = True

        branches = summary.get("num_branches", 0)
        covered_branches = summary.get("covered_branches", 0)
        branch_ratio = (covered_branches / branches) if branches > 0 else 1.0

        module_results.append(
            {
                "module": prod_file,
                "status": "PASSED" if passed else "FAILED",
                "statements": stmts,
                "covered": covered,
                "ratio": ratio,
                "branches": branches,
                "covered_branches": covered_branches,
                "branch_ratio": branch_ratio,
            }
        )

    all_passed = overall_passed and not any_module_failed

    report = {
        "passed": all_passed,
        "threshold": threshold,
        "overall": {
            "passed": overall_passed,
            "statements": total_stmts,
            "covered": total_covered,
            "ratio": overall_ratio,
            "branches": total_branches,
            "covered_branches": total_covered_branches,
            "branch_ratio": overall_branch_ratio,
        },
        "modules": module_results,
    }

    return all_passed, report


def format_report(report: dict[str, Any]) -> str:
    """Format verification report into an operational CLI summary."""
    lines: list[str] = []
    overall = report["overall"]
    threshold_pct = report["threshold"] * 100

    lines.append("\n" + "=" * 80)
    lines.append(f"COVERAGE QUALITY GATE: {'PASSED' if report['passed'] else 'FAILED'}")
    lines.append("=" * 80)

    # Overall summary
    overall_stmt_pct = overall["ratio"] * 100
    overall_branch_pct = overall["branch_ratio"] * 100
    status_tag = "[PASS]" if overall["passed"] else "[FAIL]"
    lines.append(
        f"Overall Statement Coverage: {overall_stmt_pct:6.2f}% ({overall['covered']}/{overall['statements']})  {status_tag}  (threshold >= {threshold_pct:.2f}%)"
    )
    lines.append(
        f"Overall Branch Diagnostic:  {overall_branch_pct:6.2f}% ({overall['covered_branches']}/{overall['branches']})  [DIAGNOSTIC]"
    )
    lines.append("-" * 80)
    lines.append(
        f"{'Module':<40} {'Statements':<14} {'Stmt %':<10} {'Status':<12} {'Branch %':<10}"
    )
    lines.append("-" * 80)

    for m in report["modules"]:
        mod_name = m["module"]
        status = m["status"]
        if status == "DEFERRED":
            lines.append(f"{mod_name:<40} {'-':<14} {'-':<10} {'[DEFERRED]':<12} {'-':<10}")
        elif status in {"MISSING_DATA", "INVALID_METRICS"}:
            lines.append(f"{mod_name:<40} {'0/0':<14} {'0.00%':<10} {f'[{status}]':<12} {'-':<10}")
        else:
            stmt_str = f"{m['covered']}/{m['statements']}"
            stmt_pct = f"{m['ratio'] * 100:6.2f}%"
            branch_pct = f"{m['branch_ratio'] * 100:6.2f}%" if m["branches"] > 0 else "N/A"
            tag = "[PASS]" if status == "PASSED" else "[FAIL]"
            lines.append(f"{mod_name:<40} {stmt_str:<14} {stmt_pct:<10} {tag:<12} {branch_pct:<10}")

    lines.append("=" * 80)

    # List specific failures for quick diagnosis
    failures = [m for m in report["modules"] if m["status"] not in {"PASSED", "DEFERRED"}]
    if failures or not overall["passed"]:
        lines.append("FAILURES DETECTED:")
        if not overall["passed"]:
            lines.append(
                f"  - Overall coverage {overall_stmt_pct:.2f}% is below required {threshold_pct:.2f}%"
            )
        for f in failures:
            if f["status"] == "FAILED":
                f_pct = f["ratio"] * 100
                lines.append(
                    f"  - {f['module']}: {f_pct:.2f}% ({f['covered']}/{f['statements']}) is below required {threshold_pct:.2f}%"
                )
            elif f["status"] == "MISSING_DATA":
                lines.append(
                    f"  - {f['module']}: Missing from coverage artifact (untested production file)"
                )
            elif f["status"] == "INVALID_METRICS":
                lines.append(
                    f"  - {f['module']}: Coverage entry contains invalid or missing metrics"
                )
        lines.append("=" * 80 + "\n")
    else:
        lines.append("ALL PRODUCTION MODULES MEET OR EXCEED QUALITY THRESHOLD (>= 90.00%)\n")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify coverage data meets production standards.")
    parser.add_argument(
        "coverage_file",
        nargs="?",
        default="coverage.json",
        help="Path to coverage.json file (default: coverage.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Required minimum statement coverage ratio (default: 0.90)",
    )
    parser.add_argument(
        "--app-dir",
        default="app",
        help="Path to app source directory (default: app)",
    )

    args = parser.parse_args()

    coverage_file = Path(args.coverage_file)
    app_dir = Path(args.app_dir)
    repo_root = app_dir.parent.resolve()

    try:
        passed, report = verify_coverage(
            coverage_file=coverage_file,
            app_dir=app_dir,
            repo_root=repo_root,
            threshold=args.threshold,
        )
        print(format_report(report))
        return 0 if passed else 1
    except CoverageVerificationError as e:
        print(f"\n[ERROR] Coverage verification failed: {e}\n", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"\n[UNEXPECTED ERROR] {e}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
