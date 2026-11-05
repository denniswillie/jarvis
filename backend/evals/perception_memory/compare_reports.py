from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import ArchitectureEvaluationReport, compare_to_baseline
from .models import BaselineReport


def load_report(path: Path) -> BaselineReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "report" in payload:
        return ArchitectureEvaluationReport.model_validate(payload).report
    return BaselineReport.model_validate(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two perception/memory reports")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    assessment = compare_to_baseline(
        load_report(args.current),
        load_report(args.baseline),
    )
    print(assessment.model_dump_json(indent=2))
    if not assessment.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
