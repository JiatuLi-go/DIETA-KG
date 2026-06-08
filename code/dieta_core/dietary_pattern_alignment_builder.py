#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dietary_pattern_alignment.py

基于官方来源构建 DietaryPattern 对齐资源：
- MeSH descriptor lookup/details API
- OLS4 (ONS ontology) API

输出：
- grounding_data/dietary_pattern/dietary_pattern_reference_terms.csv
- grounding_data/dietary_pattern/dietary_pattern_reference_terms.json
- grounding_data/dietary_pattern/dietary_pattern_project_term_alignment.csv
- grounding_data/dietary_pattern/dietary_pattern_project_term_alignment.json
- grounding_data/dietary_pattern/dietary_pattern_manual_review.csv
- grounding_data/dietary_pattern/dietary_pattern_source_provenance.json

说明：
- 仅纳入可由官方 API 核实的术语，不造假。
- 项目中无法稳妥对齐的 DietaryPattern 术语一律进入 manual_review。
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "grounding_data" / "dietary_pattern"
FD_GROUNDING_JSON = ROOT / "experiments" / "GCDI_KG_FD_Test" / "results" / "grounding_alignment" / "grounding_alignment_latest.json"

MESH_API = "https://id.nlm.nih.gov/mesh/lookup"
OLS4_API = "https://www.ebi.ac.uk/ols4/api"
REQUEST_TIMEOUT = 30


MESH_LABELS = [
    "Diet, Mediterranean",
    "Dietary Approaches To Stop Hypertension",
    "Diet, Ketogenic",
    "Diet, Carbohydrate-Restricted",
    "Diet, High-Protein Low-Carbohydrate",
    "Diet, Protein-Restricted",
    "Diet, Fat-Restricted",
    "Diet, Sodium-Restricted",
    "Diet, High-Fat",
    "Diet, Vegetarian",
    "Diet, Vegan",
    "Diet, Western",
    "Diet, Paleolithic",
    "Diet, Healthy",
    "Diet, Diabetic",
    "Intermittent Fasting",
    "Diet, Reducing",
]

ONS_IDS = [
    "ONS_1000038",  # mediterranean diet
    "ONS_1000037",  # DASH diet
    "ONS_1000036",  # ketogenic diet
    "ONS_1000020",  # vegetarian diet
    "ONS_1000021",  # vegan diet
    "ONS_1000012",  # intermittent fasting
    "ONS_1000013",  # time-restricted feeding
    "ONS_1000042",  # low fat diet
    "ONS_1000040",  # low-carb high-fat etc
    "ONS_1000039",  # high-carbohydrate diet
    "ONS_1000065",  # globalized diet (western diet synonym)
]


@dataclass
class SourceRecord:
    source: str
    source_id: str
    source_url: str
    official_label: str
    canonical_label: str
    aliases: list[str]
    curation_bucket: str
    raw_payload: dict[str, Any]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = text.replace("/", " ")
    text = text.replace(",", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9()\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_parenthetical(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize_terminal_diets(text: str) -> str:
    return re.sub(r"\bdiets\b", "diet", text, flags=re.IGNORECASE)


def title_case_loose(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split())


def choose_canonical_label(official_label: str, aliases: list[str]) -> str:
    candidates = []
    for term in aliases:
        t = term.strip()
        if not t:
            continue
        if "," not in t and any(
            suffix in t.lower() for suffix in [" diet", " fasting", " feeding", " pattern", " plan"]
        ):
            candidates.append(t)
    if candidates:
        # Prefer shorter but informative human-readable labels.
        return sorted(candidates, key=lambda x: (len(x), x.lower()))[0]
    return official_label


def bucket_for_label(label: str) -> str:
    lower = label.lower()
    if any(x in lower for x in ["fasting", "feeding"]):
        return "timing_pattern"
    if any(x in lower for x in ["restricted", "low", "high", "fat", "protein", "sodium", "carbohydrate", "reducing"]):
        return "composition_or_restriction"
    return "named_or_generic_pattern"


def fetch_mesh_record(label: str) -> SourceRecord:
    q = requests.get(
        f"{MESH_API}/descriptor",
        params={"label": label, "match": "exact", "limit": 10},
        timeout=REQUEST_TIMEOUT,
    )
    q.raise_for_status()
    hits = q.json()
    if not hits:
        raise ValueError(f"MeSH exact label not found: {label}")

    descriptor = hits[0]["resource"].rsplit("/", 1)[-1]
    d = requests.get(
        f"{MESH_API}/details",
        params={"descriptor": descriptor},
        timeout=REQUEST_TIMEOUT,
    )
    d.raise_for_status()
    details = d.json()
    aliases = [term["label"] for term in details.get("terms", [])]
    official_label = hits[0]["label"]
    canonical_label = choose_canonical_label(official_label, aliases)
    return SourceRecord(
        source="MeSH",
        source_id=f"MeSH:{descriptor}",
        source_url=f"https://id.nlm.nih.gov/mesh/{descriptor}",
        official_label=official_label,
        canonical_label=canonical_label,
        aliases=aliases,
        curation_bucket=bucket_for_label(canonical_label),
        raw_payload=details,
    )


def fetch_ons_record(short_form: str) -> SourceRecord:
    q = requests.get(
        f"{OLS4_API}/ontologies/ons/terms",
        params={"short_form": short_form},
        timeout=REQUEST_TIMEOUT,
    )
    q.raise_for_status()
    terms = q.json().get("_embedded", {}).get("terms", [])
    if not terms:
        raise ValueError(f"ONS short_form not found: {short_form}")
    term = terms[0]
    aliases = [term.get("label", "")] + term.get("synonyms", [])
    aliases = [a for a in aliases if a]
    canonical_label = choose_canonical_label(term["label"], aliases)
    return SourceRecord(
        source="ONS",
        source_id=term["obo_id"],
        source_url=term["iri"],
        official_label=term["label"],
        canonical_label=canonical_label,
        aliases=aliases,
        curation_bucket=bucket_for_label(canonical_label),
        raw_payload=term,
    )


def concept_key_from_record(record: SourceRecord) -> str:
    normalized_aliases = {normalize_text(a): a for a in [record.canonical_label, *record.aliases]}
    for preferred in [
        normalize_text(record.canonical_label),
        normalize_text(record.official_label),
    ]:
        if preferred in normalized_aliases:
            return preferred
    return normalize_text(record.canonical_label)


def merge_source_records(records: list[SourceRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[concept_key_from_record(record)].append(record)

    concepts = []
    for idx, (key, group) in enumerate(sorted(grouped.items()), start=1):
        preferred = next((g for g in group if g.source == "MeSH"), group[0])
        all_aliases = []
        seen_aliases = set()
        for g in group:
            for alias in [g.official_label, g.canonical_label, *g.aliases]:
                if alias and alias not in seen_aliases:
                    seen_aliases.add(alias)
                    all_aliases.append(alias)

        concepts.append(
            {
                "concept_id": f"DPAT:{idx:03d}",
                "canonical_label": preferred.canonical_label,
                "concept_key": key,
                "preferred_source": preferred.source,
                "preferred_source_id": preferred.source_id,
                "preferred_source_url": preferred.source_url,
                "curation_bucket": preferred.curation_bucket,
                "all_aliases": all_aliases,
                "sources": [
                    {
                        "source": g.source,
                        "source_id": g.source_id,
                        "source_url": g.source_url,
                        "official_label": g.official_label,
                        "aliases": g.aliases,
                    }
                    for g in group
                ],
            }
        )
    return concepts


def extract_project_terms() -> list[str]:
    if not FD_GROUNDING_JSON.exists():
        raise FileNotFoundError(FD_GROUNDING_JSON)
    with open(FD_GROUNDING_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms = []
    for triple in data.get("aligned_triples", []):
        if triple.get("subject_type") == "DietaryPattern":
            terms.append(triple.get("subject", "").strip())
        if triple.get("object_type") == "DietaryPattern":
            terms.append(triple.get("object", "").strip())
    return sorted({t for t in terms if t}, key=str.lower)


def build_alias_index(concepts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for concept in concepts:
        for alias in concept["all_aliases"]:
            norm = normalize_text(alias)
            index[norm] = {
                "concept": concept,
                "matched_alias": alias,
                "match_rule": "normalized_alias",
            }
    return index


def resolve_project_term(term: str, alias_index: dict[str, dict[str, Any]], concepts: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [
        ("normalized_raw", term),
        ("normalized_parenthetical_removed", remove_parenthetical(term)),
        ("normalized_terminal_diets_singularized", singularize_terminal_diets(term)),
        (
            "normalized_parenthetical_removed_then_singularized",
            singularize_terminal_diets(remove_parenthetical(term)),
        ),
    ]

    seen_norm = set()
    for rule, variant_text in variants:
        norm = normalize_text(variant_text)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        if norm in alias_index:
            match = alias_index[norm]
            concept = match["concept"]
            return {
                "observed_term": term,
                "observed_term_normalized": normalize_text(term),
                "mapping_status": "matched",
                "match_rule": rule,
                "matched_alias": match["matched_alias"],
                "canonical_label": concept["canonical_label"],
                "concept_id": concept["concept_id"],
                "preferred_source": concept["preferred_source"],
                "preferred_source_id": concept["preferred_source_id"],
                "preferred_source_url": concept["preferred_source_url"],
                "curation_bucket": concept["curation_bucket"],
                "all_source_ids": [s["source_id"] for s in concept["sources"]],
                "all_sources": [s["source"] for s in concept["sources"]],
                "suggested_matches": [],
            }

    scored = []
    obs_norm = normalize_text(term)
    for concept in concepts:
        score = difflib.SequenceMatcher(None, obs_norm, concept["concept_key"]).ratio()
        scored.append((score, concept["canonical_label"], concept["concept_id"]))
    scored.sort(reverse=True)
    suggestions = [
        {"canonical_label": label, "concept_id": cid, "similarity": round(score, 3)}
        for score, label, cid in scored[:3]
    ]
    return {
        "observed_term": term,
        "observed_term_normalized": obs_norm,
        "mapping_status": "manual_review",
        "match_rule": "",
        "matched_alias": "",
        "canonical_label": "",
        "concept_id": "",
        "preferred_source": "",
        "preferred_source_id": "",
        "preferred_source_url": "",
        "curation_bucket": "",
        "all_source_ids": [],
        "all_sources": [],
        "suggested_matches": suggestions,
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_records: list[SourceRecord] = []
    for label in MESH_LABELS:
        source_records.append(fetch_mesh_record(label))
    for short_form in ONS_IDS:
        source_records.append(fetch_ons_record(short_form))

    concepts = merge_source_records(source_records)
    alias_index = build_alias_index(concepts)

    project_terms = extract_project_terms()
    project_alignment = [resolve_project_term(term, alias_index, concepts) for term in project_terms]
    manual_review = [row for row in project_alignment if row["mapping_status"] == "manual_review"]

    source_rows = []
    for record in source_records:
        source_rows.append(
            {
                "source": record.source,
                "source_id": record.source_id,
                "source_url": record.source_url,
                "official_label": record.official_label,
                "canonical_label": record.canonical_label,
                "curation_bucket": record.curation_bucket,
                "aliases": json.dumps(record.aliases, ensure_ascii=False),
            }
        )

    concept_rows = []
    for concept in concepts:
        concept_rows.append(
            {
                "concept_id": concept["concept_id"],
                "canonical_label": concept["canonical_label"],
                "preferred_source": concept["preferred_source"],
                "preferred_source_id": concept["preferred_source_id"],
                "preferred_source_url": concept["preferred_source_url"],
                "curation_bucket": concept["curation_bucket"],
                "all_aliases": json.dumps(concept["all_aliases"], ensure_ascii=False),
                "sources": json.dumps(concept["sources"], ensure_ascii=False),
            }
        )

    project_rows = []
    for row in project_alignment:
        project_rows.append(
            {
                "observed_term": row["observed_term"],
                "observed_term_normalized": row["observed_term_normalized"],
                "mapping_status": row["mapping_status"],
                "match_rule": row["match_rule"],
                "matched_alias": row["matched_alias"],
                "canonical_label": row["canonical_label"],
                "concept_id": row["concept_id"],
                "preferred_source": row["preferred_source"],
                "preferred_source_id": row["preferred_source_id"],
                "preferred_source_url": row["preferred_source_url"],
                "curation_bucket": row["curation_bucket"],
                "all_source_ids": json.dumps(row["all_source_ids"], ensure_ascii=False),
                "all_sources": json.dumps(row["all_sources"], ensure_ascii=False),
                "suggested_matches": json.dumps(row["suggested_matches"], ensure_ascii=False),
            }
        )

    write_json(OUT_DIR / "dietary_pattern_source_provenance.json", source_rows)
    write_json(OUT_DIR / "dietary_pattern_reference_terms.json", concepts)
    write_json(OUT_DIR / "dietary_pattern_project_term_alignment.json", project_alignment)
    write_json(OUT_DIR / "dietary_pattern_manual_review.json", manual_review)

    write_csv(
        OUT_DIR / "dietary_pattern_source_provenance.csv",
        source_rows,
        ["source", "source_id", "source_url", "official_label", "canonical_label", "curation_bucket", "aliases"],
    )
    write_csv(
        OUT_DIR / "dietary_pattern_reference_terms.csv",
        concept_rows,
        ["concept_id", "canonical_label", "preferred_source", "preferred_source_id", "preferred_source_url", "curation_bucket", "all_aliases", "sources"],
    )
    write_csv(
        OUT_DIR / "dietary_pattern_project_term_alignment.csv",
        project_rows,
        [
            "observed_term",
            "observed_term_normalized",
            "mapping_status",
            "match_rule",
            "matched_alias",
            "canonical_label",
            "concept_id",
            "preferred_source",
            "preferred_source_id",
            "preferred_source_url",
            "curation_bucket",
            "all_source_ids",
            "all_sources",
            "suggested_matches",
        ],
    )
    write_csv(
        OUT_DIR / "dietary_pattern_manual_review.csv",
        [row for row in project_rows if row["mapping_status"] == "manual_review"],
        [
            "observed_term",
            "observed_term_normalized",
            "mapping_status",
            "match_rule",
            "matched_alias",
            "canonical_label",
            "concept_id",
            "preferred_source",
            "preferred_source_id",
            "preferred_source_url",
            "curation_bucket",
            "all_source_ids",
            "all_sources",
            "suggested_matches",
        ],
    )

    summary = {
        "reference_concepts": len(concepts),
        "source_records": len(source_records),
        "project_terms_total": len(project_terms),
        "project_terms_matched": sum(1 for row in project_alignment if row["mapping_status"] == "matched"),
        "project_terms_manual_review": sum(1 for row in project_alignment if row["mapping_status"] == "manual_review"),
        "output_dir": str(OUT_DIR),
    }
    write_json(OUT_DIR / "dietary_pattern_build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



