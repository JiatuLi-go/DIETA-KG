from __future__ import annotations

"""Ablation-trace support for DIETA.

This module prepares reproducible component ablations without forcing the final
viewer-facing output format to change. It creates machine-readable stage traces
that preserve vote counts, entropy, score components, thresholds and run variant
metadata.
"""

import json
import math
import os
import shutil
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class AblationConfig:
    variant_id: str = "full"
    disable_routing: bool = False
    disable_entity_consistency_filter: bool = False
    disable_relation_consistency_filter: bool = False
    disable_uncertainty_penalty: bool = False
    disable_feedback_loop: bool = False
    disable_grounding_feedback: bool = False
    threshold_overrides: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_variant(cls, variant_id: str) -> "AblationConfig":
        key = (variant_id or "full").strip().lower().replace("-", "_")
        cfg = cls(variant_id=key)
        if key in {"full", "baseline"}:
            cfg.variant_id = "full"
        elif key in {"no_uncertainty", "without_uncertainty", "wo_uncertainty"}:
            cfg.disable_uncertainty_penalty = True
        elif key in {"no_feedback", "without_feedback", "wo_feedback"}:
            cfg.disable_feedback_loop = True
            cfg.disable_grounding_feedback = True
        elif key in {"no_relation_consistency", "without_consistency", "wo_consistency"}:
            cfg.disable_relation_consistency_filter = True
            cfg.disable_entity_consistency_filter = True
        elif key in {"no_routing", "without_routing", "wo_routing"}:
            cfg.disable_routing = True
        else:
            cfg.notes = "custom variant; no automatic switches inferred"
        return cfg

    def to_env(self) -> Dict[str, str]:
        return {
            "GCDI_ABLATION_VARIANT": self.variant_id,
            "GCDI_ABLATION_DISABLE_ROUTING": str(int(self.disable_routing)),
            "GCDI_ABLATION_DISABLE_ENTITY_CONSISTENCY": str(int(self.disable_entity_consistency_filter)),
            "GCDI_ABLATION_DISABLE_RELATION_CONSISTENCY": str(int(self.disable_relation_consistency_filter)),
            "GCDI_ABLATION_DISABLE_UNCERTAINTY": str(int(self.disable_uncertainty_penalty)),
            "GCDI_ABLATION_DISABLE_FEEDBACK": str(int(self.disable_feedback_loop)),
            "GCDI_ABLATION_DISABLE_GROUNDING_FEEDBACK": str(int(self.disable_grounding_feedback)),
        }


def load_ablation_config(variant: str = "full", config_path: str | Path | None = None) -> AblationConfig:
    if config_path:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        base = AblationConfig.from_variant(data.get("variant_id", variant))
        for key, value in data.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base
    return AblationConfig.from_variant(variant)


def entropy_from_votes(raw_votes: Dict[str, Any]) -> float:
    vals = [v for v in (raw_votes or {}).values() if v]
    if not vals:
        return 0.0
    counts: Dict[str, int] = {}
    for v in vals:
        counts[str(v)] = counts.get(str(v), 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    max_entropy = math.log2(max(len(counts), 1)) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy else 0.0


def _iter_candidate_relations(core_dir: Path) -> Iterable[Dict[str, Any]]:
    path = core_dir / "results" / "candidate_extraction" / "candidate_relation_extraction_latest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for pmid, item in (data or {}).items():
        for topic, topic_result in (item.get("re_results", {}) or {}).items():
            for idx, triple in enumerate(topic_result.get("triples", []) or []):
                raw_votes = triple.get("raw_votes", {}) or {}
                rows.append({
                    "stage": "candidate_relation",
                    "pmid": pmid,
                    "topic": topic,
                    "triple_idx": idx,
                    "subject": triple.get("subject", ""),
                    "relation": triple.get("relation", ""),
                    "object": triple.get("object", ""),
                    "subject_type": triple.get("subject_type"),
                    "object_type": triple.get("object_type"),
                    "subject_confidence": triple.get("subject_confidence"),
                    "object_confidence": triple.get("object_confidence"),
                    "relation_confidence": triple.get("confidence"),
                    "votes": triple.get("votes"),
                    "raw_votes": raw_votes,
                    "vote_entropy": entropy_from_votes(raw_votes),
                })
    return rows


def _iter_quality_decisions(core_dir: Path) -> Iterable[Dict[str, Any]]:
    path = core_dir / "results" / "quality_decision" / "quality_decision_latest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for pmid, item in (data or {}).items():
        if not isinstance(item, dict):
            continue
        for idx, triple in enumerate(item.get("final_triples", []) or []):
            rows.append({
                "stage": "quality_decision",
                "pmid": pmid,
                "triple_idx": idx,
                "topic": triple.get("topic", triple.get("source_cluster", "")),
                "subject": triple.get("subject", ""),
                "relation": triple.get("relation", ""),
                "object": triple.get("object", ""),
                "p_eh": triple.get("p_eh", triple.get("confidence")),
                "p_et": triple.get("p_et", triple.get("confidence")),
                "p_relation": triple.get("p_relation", triple.get("confidence")),
                "Q_t": triple.get("Q_t"),
                "U_t": triple.get("U_t"),
                "lambda_reflect": triple.get("lambda_reflect"),
                "Q_t_penalized": triple.get("Q_t_penalized"),
                "Score_t": triple.get("Score_t"),
                "quality_tier": triple.get("quality_tier"),
                "feedback_rounds": triple.get("feedback_rounds"),
                "raw_votes": triple.get("raw_votes", {}),
            })
    return rows


def export_ablation_trace(core_dir: str | Path, config: AblationConfig) -> Dict[str, Any]:
    core = Path(core_dir)
    out_dir = core / "results" / "ablation" / config.variant_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_rows: List[Dict[str, Any]] = []
    trace_rows.extend(list(_iter_candidate_relations(core)))
    trace_rows.extend(list(_iter_quality_decisions(core)))
    for row in trace_rows:
        row["variant_id"] = config.variant_id
        row["disable_uncertainty_penalty"] = config.disable_uncertainty_penalty
        row["disable_feedback_loop"] = config.disable_feedback_loop
        row["disable_relation_consistency_filter"] = config.disable_relation_consistency_filter
        row["disable_entity_consistency_filter"] = config.disable_entity_consistency_filter
        row["disable_routing"] = config.disable_routing
    jsonl = out_dir / "ablation_trace_latest.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "ablation_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"variant_id": config.variant_id, "trace_rows": len(trace_rows), "trace_jsonl": str(jsonl)}


def snapshot_stage_outputs(core_dir: str | Path, config: AblationConfig, stage_name: str) -> Dict[str, Any]:
    """Copy current latest outputs into a per-variant audit folder.

    This keeps final legacy file names unchanged while making ablation reruns easy
    to compare afterwards.
    """
    core = Path(core_dir)
    dest = core / "results" / "ablation" / config.variant_id / "stage_outputs" / stage_name
    dest.mkdir(parents=True, exist_ok=True)
    patterns = [
        "results/semantic_triage/semantic_triage_latest.xlsx",
        "results/seed_entity/seed_entity_latest.json",
        "results/candidate_extraction/candidate_entity_extraction_latest.json",
        "results/candidate_extraction/candidate_entity_extraction_latest.xlsx",
        "results/candidate_extraction/candidate_relation_extraction_latest.json",
        "results/candidate_extraction/candidate_relation_extraction_latest.xlsx",
        "results/quality_decision/quality_decision_latest.json",
        "results/quality_decision/quality_decision_latest.xlsx",
        "results/grounding_alignment/grounding_alignment_latest.json",
        "results/grounding_alignment/grounding_alignment_latest.xlsx",
    ]
    copied = []
    for rel in patterns:
        src = core / rel
        if src.exists():
            dst = dest / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return {"stage": stage_name, "copied": copied}
