#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from dieta_agentic.output_contracts import normalize_grounding_final, normalize_quality_step4, write_contract_report


def main():
    parser = argparse.ArgumentParser(description="Normalize DIETA published outputs to legacy step4/step5 contracts.")
    parser.add_argument("--core-dir", default=str(Path(__file__).resolve().parent / "dieta_core"))
    args = parser.parse_args()
    report = {
        "quality_step4": normalize_quality_step4(args.core_dir),
        "grounding_step5": normalize_grounding_final(args.core_dir),
        "contract_report": write_contract_report(args.core_dir),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
