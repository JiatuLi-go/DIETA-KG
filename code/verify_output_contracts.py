#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from dieta_agentic.output_contracts import check_all_contracts, write_contract_report, CONTRACTS


def main():
    parser = argparse.ArgumentParser(description='Verify DIETA output contracts without loading full spreadsheets.')
    parser.add_argument('--core-dir', default=str(Path(__file__).resolve().parent / 'dieta_core'))
    parser.add_argument('--write-report', action='store_true')
    parser.add_argument('--show-contracts', action='store_true')
    args = parser.parse_args()
    if args.show_contracts:
        print(json.dumps(CONTRACTS, ensure_ascii=False, indent=2))
        return
    report = write_contract_report(args.core_dir) if args.write_report else check_all_contracts(args.core_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    failed = [c for c in report.get('checks', []) if c.get('status') == 'fail']
    raise SystemExit(1 if failed else 0)

if __name__ == '__main__':
    main()
