#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import shutil

REMOVE_DIRS = ['results/llm_cache', '__pycache__', '.pytest_cache']
REMOVE_PATTERNS = ['*.env', '.env', '*api_key*', '*secret*', '*.faiss', '*.pkl', '*.npy', '*.xlsx', '*.docx', '*.pdf']

def main():
    base = Path(__file__).resolve().parent
    for d in REMOVE_DIRS:
        for p in base.rglob(d):
            if p.exists(): shutil.rmtree(p, ignore_errors=True)
    for pat in REMOVE_PATTERNS:
        for p in base.rglob(pat):
            if p.name in {'README.md'}: continue
            try: p.unlink()
            except Exception: pass
    print('Sanitized release tree. Review manually before upload.')

if __name__ == '__main__':
    main()
