#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from dieta_agentic.ablation import load_ablation_config, export_ablation_trace


def main():
    parser = argparse.ArgumentParser(description="Export DIETA ablation trace from existing intermediate results.")
    parser.add_argument("--core-dir", default=str(Path(__file__).resolve().parent / "dieta_core"))
    parser.add_argument("--variant", default="full")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_ablation_config(args.variant, args.config)
    report = export_ablation_trace(args.core_dir, cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
