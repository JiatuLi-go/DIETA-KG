from __future__ import annotations

"""Output-format contracts for DIETA runtime export layer.

This module is intentionally placed in the runtime wrapper rather than inside the
scientific core. It does not change candidate generation, RUQA scoring or
standardization logic. It only normalizes published tabular exports so that
legacy downstream scripts and paper figures keep the same file paths and columns.
"""

import json
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

GROUNDING_FINAL_COLUMNS = [
    "pmid", "subject", "relation", "object", "source_cluster", "quality_tier", "Score_t",
    "head_status", "head_standard", "head_similarity", "tail_status", "tail_standard",
    "tail_similarity", "needs_feedback", "feedback_completed",
]

QUALITY_DECISION_COLUMNS = [
    # Locked to the original quality_decision_agent.py export order.
    # Do not reorder: downstream evaluation notebooks and figure scripts rely on this layout.
    "pmid", "subject", "subject_type", "relation", "object", "object_type",
    "P_entity_head", "P_entity_tail", "P_relation", "Q_t", "U_t", "Score_t",
    "quality_tier", "lambda_reflect", "Q_t_penalized", "feedback_rounds", "source_cluster",
]

CANDIDATE_ENTITY_COLUMNS = ["pmid", "abstract", "topic", "entity", "type", "confidence", "votes"]

CANDIDATE_RELATION_COLUMNS = [
    "pmid", "topic", "subject", "relation", "object", "confidence", "votes",
    "subject_confidence", "object_confidence", "abstract",
]

CONTRACTS = {
    "grounding_alignment_final": GROUNDING_FINAL_COLUMNS,
    "quality_decision_step4": QUALITY_DECISION_COLUMNS,
    "candidate_entity": CANDIDATE_ENTITY_COLUMNS,
    "candidate_relation": CANDIDATE_RELATION_COLUMNS,
}


@dataclass
class ContractCheck:
    name: str
    path: str
    exists: bool
    expected_columns: List[str]
    observed_columns: List[str]
    status: str
    message: str = ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _alignment_status(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict) and "value" in value:
        return str(value.get("value", ""))
    return str(value)


def grounding_rows_from_json(json_path: str | Path) -> List[Dict[str, Any]]:
    """Build the legacy step5 tabular rows from grounding_alignment_latest.json."""
    p = Path(json_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    triples = data.get("aligned_triples", []) if isinstance(data, dict) else []
    run_feedback_completed = bool(data.get("feedback_completed", False)) if isinstance(data, dict) else False
    rows: List[Dict[str, Any]] = []
    for triple in triples:
        head_align = triple.get("head_alignment") or {}
        tail_align = triple.get("tail_alignment") or {}
        needs_feedback = bool(triple.get("needs_feedback", False))
        feedback_completed = triple.get("feedback_completed", None)
        if feedback_completed is None:
            # Legacy after-feedback files contain this column. When the feedback
            # controller has completed at run level, every row is export-complete;
            # otherwise only rows without feedback requests are complete.
            feedback_completed = True if run_feedback_completed else not needs_feedback
        row = {
            "pmid": triple.get("pmid", ""),
            "subject": triple.get("subject", ""),
            "relation": triple.get("relation", ""),
            "object": triple.get("object", ""),
            "source_cluster": triple.get("source_cluster", triple.get("topic", "")),
            "quality_tier": triple.get("quality_tier", ""),
            "Score_t": triple.get("Score_t", ""),
            "head_status": _alignment_status(head_align.get("status", "")),
            "head_standard": head_align.get("standard_name", ""),
            "head_similarity": _safe_float(head_align.get("similarity", 0)),
            "tail_status": _alignment_status(tail_align.get("status", "")),
            "tail_standard": tail_align.get("standard_name", ""),
            "tail_similarity": _safe_float(tail_align.get("similarity", 0)),
            "needs_feedback": needs_feedback,
            "feedback_completed": bool(feedback_completed),
        }
        rows.append({col: row.get(col, "") for col in GROUNDING_FINAL_COLUMNS})
    return rows


def quality_rows_from_json(json_path: str | Path) -> List[Dict[str, Any]]:
    """Build the legacy step4 quality-decision rows from quality_decision_latest.json."""
    p = Path(json_path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for pmid, item in data.items():
            if not isinstance(item, dict):
                continue
            for triple in item.get("final_triples", []) or []:
                row = {
                    "pmid": pmid,
                    "subject": triple.get("subject", triple.get("head_entity", "")),
                    "relation": triple.get("relation", ""),
                    "object": triple.get("object", triple.get("tail_entity", "")),
                    "subject_type": triple.get("subject_type", ""),
                    "object_type": triple.get("object_type", ""),
                    "P_entity_head": triple.get("p_eh", triple.get("P_entity_head", triple.get("confidence", 0.0))),
                    "P_entity_tail": triple.get("p_et", triple.get("P_entity_tail", triple.get("confidence", 0.0))),
                    "P_relation": triple.get("p_relation", triple.get("P_relation", triple.get("confidence", 0.0))),
                    "Q_t": triple.get("Q_t", 0.0),
                    "U_t": triple.get("U_t", 0.0),
                    "Score_t": triple.get("Score_t", 0.0),
                    "quality_tier": triple.get("quality_tier", "unknown"),
                    "lambda_reflect": triple.get("lambda_reflect", 1.0),
                    "Q_t_penalized": triple.get("Q_t_penalized", 0.0),
                    "feedback_rounds": triple.get("feedback_rounds", 0),
                    "source_cluster": triple.get("source_cluster", triple.get("topic", "")),
                }
                rows.append({col: row.get(col, "") for col in QUALITY_DECISION_COLUMNS})
    return rows


def _write_xlsx_with_pandas(rows: List[Dict[str, Any]], columns: List[str], output_path: str | Path) -> None:
    """Write xlsx using the same pandas engine style as the legacy project code."""
    import pandas as pd
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_excel(out, index=False)


def normalize_grounding_final(core_dir: str | Path) -> Dict[str, Any]:
    core = Path(core_dir)
    json_path = core / "results" / "grounding_alignment" / "grounding_alignment_latest.json"
    xlsx_path = core / "results" / "grounding_alignment" / "grounding_alignment_latest.xlsx"
    rows = grounding_rows_from_json(json_path)
    if rows:
        _write_xlsx_with_pandas(rows, GROUNDING_FINAL_COLUMNS, xlsx_path)
        ts = time.strftime("%Y%m%d_%H%M%S")
        _write_xlsx_with_pandas(rows, GROUNDING_FINAL_COLUMNS, xlsx_path.parent / f"grounding_alignment_contract_{ts}.xlsx")
    return {"path": str(xlsx_path), "rows": len(rows), "columns": GROUNDING_FINAL_COLUMNS}


def normalize_quality_step4(core_dir: str | Path) -> Dict[str, Any]:
    core = Path(core_dir)
    json_path = core / "results" / "quality_decision" / "quality_decision_latest.json"
    xlsx_path = core / "results" / "quality_decision" / "quality_decision_latest.xlsx"
    rows = quality_rows_from_json(json_path)
    if rows:
        _write_xlsx_with_pandas(rows, QUALITY_DECISION_COLUMNS, xlsx_path)
        ts = time.strftime("%Y%m%d_%H%M%S")
        _write_xlsx_with_pandas(rows, QUALITY_DECISION_COLUMNS, xlsx_path.parent / f"quality_decision_contract_{ts}.xlsx")
    return {"path": str(xlsx_path), "rows": len(rows), "columns": QUALITY_DECISION_COLUMNS}


def _xlsx_header(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(p) as z:
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.findall("main:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//main:t", ns)))
        row = root.find(".//main:sheetData/main:row", ns)
        if row is None:
            return []
        values = []
        for c in row.findall("main:c", ns):
            typ = c.attrib.get("t")
            if typ == "inlineStr":
                values.append("".join(t.text or "" for t in c.findall(".//main:t", ns)))
            else:
                v = c.find("main:v", ns)
                raw = v.text if v is not None else ""
                values.append(shared[int(raw)] if typ == "s" and raw else (raw or ""))
        return values


def check_xlsx_contract(path: str | Path, expected: List[str], name: str) -> ContractCheck:
    p = Path(path)
    observed = _xlsx_header(p) if p.exists() else []
    if not p.exists():
        status = "missing"
        msg = "file not found"
    elif observed == expected:
        status = "pass"
        msg = "columns match exactly"
    else:
        status = "fail"
        msg = f"expected {expected}, observed {observed}"
    return ContractCheck(name=name, path=str(p), exists=p.exists(), expected_columns=expected, observed_columns=observed, status=status, message=msg)


def check_all_contracts(core_dir: str | Path) -> Dict[str, Any]:
    core = Path(core_dir)
    checks = [
        check_xlsx_contract(core / "results" / "grounding_alignment" / "grounding_alignment_latest.xlsx", GROUNDING_FINAL_COLUMNS, "grounding_alignment_final"),
        check_xlsx_contract(core / "results" / "quality_decision" / "quality_decision_latest.xlsx", QUALITY_DECISION_COLUMNS, "quality_decision_step4"),
        check_xlsx_contract(core / "results" / "candidate_extraction" / "candidate_entity_extraction_latest.xlsx", CANDIDATE_ENTITY_COLUMNS, "candidate_entity"),
        check_xlsx_contract(core / "results" / "candidate_extraction" / "candidate_relation_extraction_latest.xlsx", CANDIDATE_RELATION_COLUMNS, "candidate_relation"),
    ]
    return {"checks": [asdict(c) for c in checks], "passed": all(c.status in {"pass", "missing"} for c in checks)}


def write_contract_report(core_dir: str | Path) -> Dict[str, Any]:
    report = check_all_contracts(core_dir)
    out = Path(core_dir) / "results" / "contracts" / "output_contract_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
