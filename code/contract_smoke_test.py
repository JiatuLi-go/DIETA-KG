from __future__ import annotations

"""Small contract smoke test for DIETA tabular exports.

The test builds minimal Step 4 and Step 5 JSON fixtures, runs the same contract
normalizers used by the workflow, and verifies the resulting XLSX headers.
"""

import json
import tempfile
from pathlib import Path

from dieta_agentic.output_contracts import (
    GROUNDING_FINAL_COLUMNS,
    QUALITY_DECISION_COLUMNS,
    check_all_contracts,
    normalize_grounding_final,
    normalize_quality_step4,
)


def _write_fixture(base: Path) -> None:
    qdir = base / "results" / "quality_decision"
    gdir = base / "results" / "grounding_alignment"
    qdir.mkdir(parents=True, exist_ok=True)
    gdir.mkdir(parents=True, exist_ok=True)

    quality = {
        "100": {
            "final_triples": [
                {
                    "subject": "whole grain",
                    "subject_type": "Food",
                    "relation": "recommended_for",
                    "object": "type 2 diabetes",
                    "object_type": "Disease",
                    "p_eh": 1.0,
                    "p_et": 1.0,
                    "p_relation": 0.8,
                    "Q_t": 0.8,
                    "U_t": 0.0,
                    "Score_t": 0.8,
                    "quality_tier": "high",
                    "lambda_reflect": 1.0,
                    "Q_t_penalized": 0.8,
                    "feedback_rounds": 0,
                    "source_cluster": "T1",
                }
            ]
        }
    }
    grounding = {
        "aligned_triples": [
            {
                "pmid": "100",
                "subject": "whole grain",
                "relation": "recommended_for",
                "object": "type 2 diabetes",
                "source_cluster": "T1",
                "quality_tier": "high",
                "Score_t": 0.8,
                "head_alignment": {"status": "direct_hit", "standard_name": "WHOLE GRAIN", "similarity": 1.0},
                "tail_alignment": {"status": "direct_hit", "standard_name": "diabetes mellitus", "similarity": 0.89},
                "needs_feedback": False,
                "feedback_completed": True,
            }
        ],
        "feedback_requests": [],
    }
    (qdir / "quality_decision_latest.json").write_text(json.dumps(quality), encoding="utf-8")
    (gdir / "grounding_alignment_latest.json").write_text(json.dumps(grounding), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _write_fixture(base)
        normalize_quality_step4(base)
        normalize_grounding_final(base)
        report = check_all_contracts(base)
        wanted = {"grounding_alignment_final", "quality_decision_step4"}
        failed = [c for c in report["checks"] if c["name"] in wanted and c["status"] != "pass"]
        if failed:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 1
        print("Contract smoke test passed")
        print("Step4 columns:", ", ".join(QUALITY_DECISION_COLUMNS))
        print("Step5 columns:", ", ".join(GROUNDING_FINAL_COLUMNS))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
