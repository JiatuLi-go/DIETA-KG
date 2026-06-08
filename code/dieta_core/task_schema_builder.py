from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, SKOS, OWL, XSD, URIRef
from rdflib.namespace import DCTERMS


BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
ONTOLOGY_DIR = BASE_DIR / "本体"
OUTPUT_DIR = ONTOLOGY_DIR / "DIETA_task_ontology"
RESULTS_DIR = BASE_DIR / "experiments" / "GCDI_KG_FD_Test" / "results"
DIETARY_PATTERN_DIR = BASE_DIR / "grounding_data" / "dietary_pattern"


TASK_ONTOLOGY_IRI = URIRef("https://gcdi-kg.local/ontology/dieta-task-constraint-layer")
BRIDGE_ONTOLOGY_IRI = URIRef("https://gcdi-kg.local/ontology/dieta-semantic-bridge")
DIETA = Namespace("https://gcdi-kg.local/ontology/dieta#")


CLASS_NAMES = {
    "KnowledgeEntity": DIETA.KnowledgeEntity,
    "InterventionSource": DIETA.InterventionSource,
    "Food": DIETA.Food,
    "Nutrient": DIETA.Nutrient,
    "Disease": DIETA.Disease,
    "DietaryPattern": DIETA.DietaryPattern,
    "NamedDietaryPattern": DIETA.NamedDietaryPattern,
    "RestrictionDietaryPattern": DIETA.RestrictionDietaryPattern,
    "TimingDietaryPattern": DIETA.TimingDietaryPattern,
    "DietaryPatternCandidate": DIETA.DietaryPatternCandidate,
    "TopicCluster": DIETA.TopicCluster,
}


ANNOTATION_PROPS = {
    "runtimeGroundingChannel": DIETA.runtimeGroundingChannel,
    "modelSupport": DIETA.modelSupport,
    "preferredSource": DIETA.preferredSource,
    "preferredSourceId": DIETA.preferredSourceId,
    "preferredSourceUrl": DIETA.preferredSourceUrl,
    "sourceFile": DIETA.sourceFile,
    "curationBucket": DIETA.curationBucket,
    "mappingStatus": DIETA.mappingStatus,
    "suggestedMatch": DIETA.suggestedMatch,
    "routerOrder": DIETA.routerOrder,
    "fdDocumentCount": DIETA.fdDocumentCount,
    "fdGroundedTripleCount": DIETA.fdGroundedTripleCount,
    "schemaOriginPrompt": DIETA.schemaOriginPrompt,
    "aliasText": DIETA.aliasText,
    "topicId": DIETA.topicId,
    "topicName": DIETA.topicName,
}


OBJECT_PROPS = {
    "recommended_for": DIETA.recommended_for,
    "avoid_for": DIETA.avoid_for,
    "beneficial": DIETA.beneficial,
    "harmful": DIETA.harmful,
    "has_nutrient": DIETA.has_nutrient,
    "provides_nutrient": DIETA.provides_nutrient,
    "includes_food": DIETA.includes_food,
    "hasHeadType": DIETA.hasHeadType,
    "hasTailType": DIETA.hasTailType,
    "allowsRelation": DIETA.allowsRelation,
    "usesModelForType": DIETA.usesModelForType,
}


RELATION_FALLBACK_DEFINITIONS = {
    "recommended_for": "The source entity has a preventive, alleviating, or improving effect on the disease and is therefore recommended.",
    "avoid_for": "The source entity has an inducing, aggravating, or worsening effect on the disease and should be avoided.",
    "beneficial": "The nutrient exerts a beneficial or protective effect on the disease.",
    "harmful": "The nutrient aggravates, induces, or worsens the disease.",
    "has_nutrient": "The food contains or provides the nutrient.",
    "provides_nutrient": "The dietary pattern provides or is rich in the nutrient.",
    "includes_food": "The dietary pattern explicitly includes or is characterized by the food.",
}


ENTITY_RUNTIME_CHANNELS = {
    "Food": "USDA FoodData Central triplets + FoodOn semantic bridge",
    "Nutrient": "USDA FoodData Central triplets + CDNO nutrient bridge",
    "Disease": "HumanDO runtime grounding + ICD10CM chronic disease subset bridge",
    "DietaryPattern": "MeSH/ONS curated lexicon + open-world retention",
}


ENTITY_SOURCE_FILES = {
    "Food": "本体/food_nutrient_triplets_with_portion_attribute.csv; 本体/extracted_FOODON_00002403.owl",
    "Nutrient": "本体/food_nutrient_triplets_with_portion_attribute.csv; 本体/extracted_CDNO_0000001.owl",
    "Disease": "本体/HumanDO.json; 本体/extracted_elderly_chronic.owl",
    "DietaryPattern": "grounding_data/dietary_pattern/*.csv; curated MeSH/ONS lexicon",
}


EXTERNAL_SOURCE_ROWS = [
    {
        "branch_name": "Food semantic bridge",
        "entity_type": "Food",
        "source_name": "FoodOn subset",
        "local_file": "本体/extracted_FOODON_00002403.owl",
        "anchor_iri": "http://purl.obolibrary.org/obo/FOODON_00002403",
        "anchor_label": "食物",
        "role": "semantic bridge for food class hierarchy",
        "status": "active_in_bridge",
        "notes": "Used to rebuild the food branch in the project ontology layer; runtime string grounding still uses USDA FoodData Central.",
    },
    {
        "branch_name": "Nutrient semantic bridge",
        "entity_type": "Nutrient",
        "source_name": "CDNO subset",
        "local_file": "本体/extracted_CDNO_0000001.owl",
        "anchor_iri": "http://purl.obolibrary.org/obo/CDNO_0000001",
        "anchor_label": "营养素",
        "role": "semantic bridge for nutrient class hierarchy",
        "status": "active_in_bridge",
        "notes": "Provides nutrient hierarchy anchors complementary to USDA nutrient records.",
    },
    {
        "branch_name": "Disease semantic bridge",
        "entity_type": "Disease",
        "source_name": "ICD10CM chronic-disease subset",
        "local_file": "本体/extracted_elderly_chronic.owl",
        "anchor_iri": "http://purl.bioontology.org/ontology/ICD10CM/E11",
        "anchor_label": "2型糖尿病（Type 2 DM）",
        "role": "task-focused diabetes/chronic-disease subset",
        "status": "active_in_bridge",
        "notes": "Acts as a disease-side ontology bridge; runtime disease grounding still uses HumanDO.",
    },
    {
        "branch_name": "Disease runtime grounding",
        "entity_type": "Disease",
        "source_name": "Human Disease Ontology",
        "local_file": "本体/HumanDO.json",
        "anchor_iri": "http://purl.obolibrary.org/obo/DOID_4",
        "anchor_label": "disease",
        "role": "runtime disease normalization",
        "status": "active_runtime",
        "notes": "Primary runtime disease grounding resource in the current implementation.",
    },
    {
        "branch_name": "Food and nutrient runtime grounding",
        "entity_type": "Food|Nutrient",
        "source_name": "USDA FoodData Central-derived triplets",
        "local_file": "本体/food_nutrient_triplets_with_portion_attribute.csv",
        "anchor_iri": "",
        "anchor_label": "task-curated food-nutrient triplets",
        "role": "runtime food and nutrient normalization",
        "status": "active_runtime",
        "notes": "Current runtime grounding for Food and Nutrient is based on the cleaned triplet resource rather than directly on FoodOn/CDNO OWL files.",
    },
    {
        "branch_name": "DietaryPattern lexicon",
        "entity_type": "DietaryPattern",
        "source_name": "MeSH + ONS curated lexicon",
        "local_file": "grounding_data/dietary_pattern/dietary_pattern_reference_terms.csv",
        "anchor_iri": "",
        "anchor_label": "curated dietary pattern concepts",
        "role": "runtime lexical grounding + ontology branch seed",
        "status": "active_runtime_and_bridge",
        "notes": "Provides the initial DietaryPattern vocabulary and candidate curation channel.",
    },
    {
        "branch_name": "Historical project ontology",
        "entity_type": "mixed",
        "source_name": "老年人膳食知识图谱.owl",
        "local_file": "本体/老年人膳食知识图谱.owl",
        "anchor_iri": "",
        "anchor_label": "historical file",
        "role": "historical artifact",
        "status": "malformed_historical_not_primary",
        "notes": "The file is malformed and was not reused as the primary source for the rebuilt ontology assets.",
    },
]


DIABETES_SAMPLE_MATCHES = [
    {
        "local_name": "Type1DiabetesMellitus",
        "label": "1型糖尿病（Type 1 DM）",
        "source_iri": "http://purl.bioontology.org/ontology/ICD10CM/E10",
    },
    {
        "local_name": "Type2DiabetesMellitus",
        "label": "2型糖尿病（Type 2 DM）",
        "source_iri": "http://purl.bioontology.org/ontology/ICD10CM/E11",
    },
    {
        "local_name": "OtherSpecifiedDiabetes",
        "label": "其他特指糖尿病",
        "source_iri": "http://purl.bioontology.org/ontology/ICD10CM/E13",
    },
]


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_topic_config() -> Dict[str, Dict[str, Any]]:
    main_module = load_module(BASE_DIR / "config.py", "gcdi_main_config")
    exp_module = load_module(BASE_DIR / "experiments" / "GCDI_KG_FD_Test" / "config.py", "gcdi_exp_config")
    if json.dumps(main_module.TOPIC_CONFIG, sort_keys=True, ensure_ascii=False) != json.dumps(
        exp_module.TOPIC_CONFIG, sort_keys=True, ensure_ascii=False
    ):
        raise RuntimeError("Main config and FD experiment config expose different TOPIC_CONFIG definitions.")
    return main_module.TOPIC_CONFIG


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_router_entity_definitions() -> Dict[str, str]:
    text = read_text(PROMPTS_DIR / "router.txt")
    entity_defs: Dict[str, str] = {}
    normalize_name = {
        "Dietary Pattern": "DietaryPattern",
    }
    for line in text.splitlines():
        if line.startswith("- "):
            name, sep, definition = line[2:].partition(":")
            if sep:
                clean_name = normalize_name.get(name.strip(), name.strip())
                entity_defs[clean_name] = definition.strip()
    return entity_defs


def extract_relation_definitions(topic_config: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    relation_defs: Dict[str, str] = {}
    inline_pattern = re.compile(r"^([a-z_]+):\s*(.+)$", re.IGNORECASE)
    block_pattern = re.compile(r"\[Relation:\s*([a-z_]+)\]\s*(.+)", re.IGNORECASE)
    for topic in topic_config:
        prompt_path = PROMPTS_DIR / f"re_{topic}_q1.txt"
        if not prompt_path.exists():
            continue
        lines = [line.strip() for line in read_text(prompt_path).splitlines()]
        for idx, line in enumerate(lines):
            if not line:
                continue
            match = inline_pattern.match(line)
            if match:
                relation, definition = match.groups()
                relation_defs.setdefault(relation, definition)
                continue
            block = block_pattern.match(line)
            if block and idx + 1 < len(lines):
                relation, _ = block.groups()
                next_line = lines[idx + 1].strip()
                if next_line and not next_line.startswith("["):
                    relation_defs.setdefault(relation, next_line)
    for relation, definition in RELATION_FALLBACK_DEFINITIONS.items():
        relation_defs.setdefault(relation, definition)
    return relation_defs


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed_topic_counts() -> Counter:
    path = RESULTS_DIR / "seed_entity" / "seed_entity_latest.json"
    data = load_json(path)
    counter: Counter = Counter()
    for item in data.values():
        for topic in item.get("matched_clusters", []):
            counter[topic] += 1
    return counter


def load_grounding_counts() -> tuple[Counter, Counter]:
    path = RESULTS_DIR / "grounding_alignment" / "grounding_alignment_latest.json"
    data = load_json(path)
    topic_counter: Counter = Counter()
    relation_counter: Counter = Counter()
    for item in data.get("aligned_triples", []):
        subj_type = item.get("subject_type")
        obj_type = item.get("object_type")
        source_cluster = item.get("source_cluster")
        relation = item.get("relation")
        if isinstance(subj_type, float) and math.isnan(subj_type):
            subj_type = None
        if isinstance(obj_type, float) and math.isnan(obj_type):
            obj_type = None
        if isinstance(source_cluster, float) and math.isnan(source_cluster):
            source_cluster = None
        if subj_type in {"Food", "Nutrient", "Disease", "DietaryPattern"} and obj_type in {
            "Food",
            "Nutrient",
            "Disease",
            "DietaryPattern",
        }:
            if relation:
                relation_counter[relation] += 1
            if source_cluster in {"T1", "T2", "T3", "T4", "T5", "T6"}:
                topic_counter[source_cluster] += 1
    return topic_counter, relation_counter


def load_dietary_pattern_resources() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ref_path = DIETARY_PATTERN_DIR / "dietary_pattern_reference_terms.csv"
    manual_path = DIETARY_PATTERN_DIR / "dietary_pattern_manual_review.csv"
    with ref_path.open("r", encoding="utf-8-sig", newline="") as f:
        reference_rows = list(csv.DictReader(f))
    with manual_path.open("r", encoding="utf-8-sig", newline="") as f:
        manual_rows = list(csv.DictReader(f))
    return reference_rows, manual_rows


def slugify(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return slug or "item"


def parse_json_field(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def build_entity_rows(
    topic_config: Dict[str, Dict[str, Any]],
    router_defs: Dict[str, str],
) -> List[Dict[str, Any]]:
    topic_support: Dict[str, List[str]] = {entity: [] for entity in ["Food", "Nutrient", "Disease", "DietaryPattern"]}
    model_support: Dict[str, List[str]] = {entity: [] for entity in ["Food", "Nutrient", "Disease", "DietaryPattern"]}

    for topic_id, info in topic_config.items():
        topic_support[info["head_type"]].append(topic_id)
        topic_support[info["tail_type"]].append(topic_id)
        for entity_type in info["need_models"]:
            model_support[entity_type].append(topic_id)

    rows = []
    for entity_type in ["Food", "Nutrient", "Disease", "DietaryPattern"]:
        rows.append(
            {
                "entity_type": entity_type,
                "router_definition": router_defs.get(entity_type, ""),
                "supported_topics": "|".join(sorted(set(topic_support[entity_type]))),
                "model_supported_topics": "|".join(sorted(set(model_support[entity_type]))),
                "runtime_grounding_channel": ENTITY_RUNTIME_CHANNELS[entity_type],
                "primary_source_files": ENTITY_SOURCE_FILES[entity_type],
            }
        )
    return rows


def build_relation_rows(
    topic_config: Dict[str, Dict[str, Any]],
    relation_defs: Dict[str, str],
    grounding_relation_counts: Counter,
) -> List[Dict[str, Any]]:
    relation_topics: Dict[str, List[str]] = {}
    head_types: Dict[str, set[str]] = {}
    tail_types: Dict[str, set[str]] = {}
    prompt_origin: Dict[str, List[str]] = {}
    for topic_id, info in topic_config.items():
        prompt_file = f"prompts/re_{topic_id}_q1.txt"
        for relation in info["relations"]:
            relation_topics.setdefault(relation, []).append(topic_id)
            head_types.setdefault(relation, set()).add(info["head_type"])
            tail_types.setdefault(relation, set()).add(info["tail_type"])
            prompt_origin.setdefault(relation, []).append(prompt_file)
    rows = []
    for relation in [
        "recommended_for",
        "avoid_for",
        "beneficial",
        "harmful",
        "has_nutrient",
        "includes_food",
        "provides_nutrient",
    ]:
        rows.append(
            {
                "relation": relation,
                "definition": relation_defs.get(relation, RELATION_FALLBACK_DEFINITIONS[relation]),
                "allowed_head_types": "|".join(sorted(head_types.get(relation, set()))),
                "allowed_tail_types": "|".join(sorted(tail_types.get(relation, set()))),
                "topic_ids": "|".join(sorted(relation_topics.get(relation, []))),
                "schema_origin_prompts": "|".join(sorted(prompt_origin.get(relation, []))),
                "fd_grounded_triple_count": grounding_relation_counts.get(relation, 0),
            }
        )
    # Implementation note for T6 prompt wording vs. formal config.
    rows.append(
        {
            "relation": "recommended_for_or_avoid_for",
            "definition": "Prompt-surface label used in re_T6_q1.txt; normalized at the formal schema layer into recommended_for and avoid_for.",
            "allowed_head_types": "DietaryPattern",
            "allowed_tail_types": "Disease",
            "topic_ids": "T6",
            "schema_origin_prompts": "prompts/re_T6_q1.txt",
            "fd_grounded_triple_count": 0,
        }
    )
    return rows


def build_topic_rows(
    topic_config: Dict[str, Dict[str, Any]],
    seed_topic_counts: Counter,
    grounding_topic_counts: Counter,
) -> List[Dict[str, Any]]:
    rows = []
    for idx, topic_id in enumerate(["T1", "T2", "T3", "T4", "T5", "T6"], start=1):
        info = topic_config[topic_id]
        rows.append(
            {
                "topic_id": topic_id,
                "router_order": idx,
                "topic_name": info["name"],
                "head_type": info["head_type"],
                "relation_space": "|".join(info["relations"]),
                "tail_type": info["tail_type"],
                "need_models": "|".join(info["need_models"]) if info["need_models"] else "none",
                "ner_prompt": f"prompts/ner_{topic_id}_p1.txt",
                "re_prompt": f"prompts/re_{topic_id}_q1.txt",
                "fd_document_count": seed_topic_counts.get(topic_id, 0),
                "fd_grounded_triple_count": grounding_topic_counts.get(topic_id, 0),
            }
        )
    return rows


def build_legal_pattern_rows(topic_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in topic_rows:
        for relation in row["relation_space"].split("|"):
            rows.append(
                {
                    "topic_id": row["topic_id"],
                    "head_type": row["head_type"],
                    "relation": relation,
                    "tail_type": row["tail_type"],
                    "schema_status": "allowed",
                    "ner_prompt": row["ner_prompt"],
                    "re_prompt": row["re_prompt"],
                }
            )
    return rows


def graph_with_prefixes() -> Graph:
    g = Graph()
    g.bind("dieta", DIETA)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    return g


def add_annotation_property_declarations(g: Graph) -> None:
    for prop in ANNOTATION_PROPS.values():
        g.add((prop, RDF.type, OWL.AnnotationProperty))
    g.add((OBJECT_PROPS["hasHeadType"], RDF.type, OWL.ObjectProperty))
    g.add((OBJECT_PROPS["hasTailType"], RDF.type, OWL.ObjectProperty))
    g.add((OBJECT_PROPS["allowsRelation"], RDF.type, OWL.ObjectProperty))
    g.add((OBJECT_PROPS["usesModelForType"], RDF.type, OWL.ObjectProperty))
    g.add((DIETA.routerOrderValue, RDF.type, OWL.DatatypeProperty))
    g.add((DIETA.fdDocumentCountValue, RDF.type, OWL.DatatypeProperty))
    g.add((DIETA.fdGroundedTripleCountValue, RDF.type, OWL.DatatypeProperty))


def add_literal(g: Graph, subject: URIRef, predicate: URIRef, value: Any, datatype: URIRef | None = None) -> None:
    if value is None or value == "":
        return
    if datatype is not None:
        g.add((subject, predicate, Literal(value, datatype=datatype)))
    else:
        g.add((subject, predicate, Literal(value)))


def build_task_ontology_graph(
    entity_rows: List[Dict[str, Any]],
    relation_rows: List[Dict[str, Any]],
    topic_rows: List[Dict[str, Any]],
) -> Graph:
    g = graph_with_prefixes()
    add_annotation_property_declarations(g)

    g.add((TASK_ONTOLOGY_IRI, RDF.type, OWL.Ontology))
    add_literal(g, TASK_ONTOLOGY_IRI, DCTERMS.title, "DIETA task constraint layer ontology")
    add_literal(
        g,
        TASK_ONTOLOGY_IRI,
        DCTERMS.description,
        "Operational task ontology extracted from DIETA prompts, config and FD experiment outputs. It explicitly defines entity classes, legal relation space and topic clusters.",
    )
    add_literal(g, TASK_ONTOLOGY_IRI, OWL.versionInfo, datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    # Core classes
    for class_name, iri in CLASS_NAMES.items():
        g.add((iri, RDF.type, OWL.Class))
        add_literal(g, iri, RDFS.label, class_name)

    g.add((CLASS_NAMES["Food"], RDFS.subClassOf, CLASS_NAMES["KnowledgeEntity"]))
    g.add((CLASS_NAMES["Nutrient"], RDFS.subClassOf, CLASS_NAMES["KnowledgeEntity"]))
    g.add((CLASS_NAMES["Disease"], RDFS.subClassOf, CLASS_NAMES["KnowledgeEntity"]))
    g.add((CLASS_NAMES["DietaryPattern"], RDFS.subClassOf, CLASS_NAMES["KnowledgeEntity"]))
    g.add((CLASS_NAMES["Food"], RDFS.subClassOf, CLASS_NAMES["InterventionSource"]))
    g.add((CLASS_NAMES["DietaryPattern"], RDFS.subClassOf, CLASS_NAMES["InterventionSource"]))
    g.add((CLASS_NAMES["NamedDietaryPattern"], RDFS.subClassOf, CLASS_NAMES["DietaryPattern"]))
    g.add((CLASS_NAMES["RestrictionDietaryPattern"], RDFS.subClassOf, CLASS_NAMES["DietaryPattern"]))
    g.add((CLASS_NAMES["TimingDietaryPattern"], RDFS.subClassOf, CLASS_NAMES["DietaryPattern"]))
    g.add((CLASS_NAMES["DietaryPatternCandidate"], RDFS.subClassOf, CLASS_NAMES["DietaryPattern"]))

    entity_by_name = {row["entity_type"]: CLASS_NAMES[row["entity_type"]] for row in entity_rows}
    for row in entity_rows:
        iri = entity_by_name[row["entity_type"]]
        add_literal(g, iri, RDFS.comment, row["router_definition"])
        add_literal(g, iri, ANNOTATION_PROPS["runtimeGroundingChannel"], row["runtime_grounding_channel"])
        add_literal(g, iri, ANNOTATION_PROPS["modelSupport"], row["model_supported_topics"] or "prompt_only_or_none")
        add_literal(g, iri, ANNOTATION_PROPS["sourceFile"], row["primary_source_files"])

    # Relation properties
    relation_domains = {
        "recommended_for": CLASS_NAMES["InterventionSource"],
        "avoid_for": CLASS_NAMES["InterventionSource"],
        "beneficial": CLASS_NAMES["Nutrient"],
        "harmful": CLASS_NAMES["Nutrient"],
        "has_nutrient": CLASS_NAMES["Food"],
        "provides_nutrient": CLASS_NAMES["DietaryPattern"],
        "includes_food": CLASS_NAMES["DietaryPattern"],
    }
    relation_ranges = {
        "recommended_for": CLASS_NAMES["Disease"],
        "avoid_for": CLASS_NAMES["Disease"],
        "beneficial": CLASS_NAMES["Disease"],
        "harmful": CLASS_NAMES["Disease"],
        "has_nutrient": CLASS_NAMES["Nutrient"],
        "provides_nutrient": CLASS_NAMES["Nutrient"],
        "includes_food": CLASS_NAMES["Food"],
    }

    for row in relation_rows:
        relation = row["relation"]
        if relation not in OBJECT_PROPS:
            continue
        iri = OBJECT_PROPS[relation]
        g.add((iri, RDF.type, OWL.ObjectProperty))
        add_literal(g, iri, RDFS.label, relation)
        add_literal(g, iri, RDFS.comment, row["definition"])
        g.add((iri, RDFS.domain, relation_domains[relation]))
        g.add((iri, RDFS.range, relation_ranges[relation]))
        add_literal(g, iri, ANNOTATION_PROPS["schemaOriginPrompt"], row["schema_origin_prompts"])
        add_literal(g, iri, ANNOTATION_PROPS["fdGroundedTripleCount"], row["fd_grounded_triple_count"], XSD.integer)

    # Topic individuals
    for row in topic_rows:
        iri = DIETA[f"{row['topic_id']}_{row['topic_name']}"]
        g.add((iri, RDF.type, CLASS_NAMES["TopicCluster"]))
        add_literal(g, iri, RDFS.label, f"{row['topic_id']} {row['topic_name']}")
        add_literal(g, iri, RDFS.comment, f"{row['head_type']} -> {row['tail_type']} with relation space {row['relation_space']}")
        add_literal(g, iri, ANNOTATION_PROPS["topicId"], row["topic_id"])
        add_literal(g, iri, ANNOTATION_PROPS["topicName"], row["topic_name"])
        add_literal(g, iri, ANNOTATION_PROPS["routerOrder"], row["router_order"], XSD.integer)
        add_literal(g, iri, ANNOTATION_PROPS["fdDocumentCount"], row["fd_document_count"], XSD.integer)
        add_literal(g, iri, ANNOTATION_PROPS["fdGroundedTripleCount"], row["fd_grounded_triple_count"], XSD.integer)
        add_literal(g, iri, ANNOTATION_PROPS["schemaOriginPrompt"], f"{row['ner_prompt']} | {row['re_prompt']}")
        g.add((iri, OBJECT_PROPS["hasHeadType"], entity_by_name[row["head_type"]]))
        g.add((iri, OBJECT_PROPS["hasTailType"], entity_by_name[row["tail_type"]]))
        for model_type in filter(None, row["need_models"].split("|")):
            if model_type in entity_by_name:
                g.add((iri, OBJECT_PROPS["usesModelForType"], entity_by_name[model_type]))
        for relation in filter(None, row["relation_space"].split("|")):
            g.add((iri, OBJECT_PROPS["allowsRelation"], OBJECT_PROPS[relation]))

    return g


def build_bridge_graph(
    task_graph: Graph,
    reference_rows: List[Dict[str, Any]],
    manual_rows: List[Dict[str, Any]],
) -> Graph:
    g = graph_with_prefixes()
    for triple in task_graph:
        g.add(triple)

    g.add((BRIDGE_ONTOLOGY_IRI, RDF.type, OWL.Ontology))
    g.add((BRIDGE_ONTOLOGY_IRI, OWL.imports, TASK_ONTOLOGY_IRI))
    add_literal(g, BRIDGE_ONTOLOGY_IRI, DCTERMS.title, "DIETA semantic bridge ontology")
    add_literal(
        g,
        BRIDGE_ONTOLOGY_IRI,
        DCTERMS.description,
        "Bridge ontology linking the DIETA task constraint layer to FoodOn, CDNO, ICD10CM subset and the curated MeSH/ONS dietary pattern lexicon.",
    )
    add_literal(g, BRIDGE_ONTOLOGY_IRI, OWL.versionInfo, datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    # Source anchors
    anchor_specs = [
        (
            "FoodOnFoodBranchAnchor",
            "FoodOn food branch anchor",
            CLASS_NAMES["Food"],
            "http://purl.obolibrary.org/obo/FOODON_00002403",
            "本体/extracted_FOODON_00002403.owl",
            "Food semantic bridge anchor reconstructed from the local FoodOn subset.",
        ),
        (
            "CDNONutrientBranchAnchor",
            "CDNO nutrient branch anchor",
            CLASS_NAMES["Nutrient"],
            "http://purl.obolibrary.org/obo/CDNO_0000001",
            "本体/extracted_CDNO_0000001.owl",
            "Nutrient semantic bridge anchor reconstructed from the local CDNO subset.",
        ),
        (
            "ICD10DiabetesSubsetAnchor",
            "ICD10CM diabetes subset anchor",
            CLASS_NAMES["Disease"],
            "http://purl.bioontology.org/ontology/ICD10CM/E11",
            "本体/extracted_elderly_chronic.owl",
            "Disease-side ontology bridge assembled from the local chronic-disease subset.",
        ),
    ]
    for local_name, label, parent, match_iri, source_file, comment in anchor_specs:
        iri = DIETA[local_name]
        g.add((iri, RDF.type, OWL.Class))
        g.add((iri, RDFS.subClassOf, parent))
        add_literal(g, iri, RDFS.label, label)
        add_literal(g, iri, RDFS.comment, comment)
        add_literal(g, iri, ANNOTATION_PROPS["sourceFile"], source_file)
        g.add((iri, SKOS.closeMatch, URIRef(match_iri)))

    for spec in DIABETES_SAMPLE_MATCHES:
        iri = DIETA[spec["local_name"]]
        g.add((iri, RDF.type, OWL.Class))
        g.add((iri, RDFS.subClassOf, CLASS_NAMES["Disease"]))
        add_literal(g, iri, RDFS.label, spec["label"])
        g.add((iri, SKOS.exactMatch, URIRef(spec["source_iri"])))
        add_literal(g, iri, ANNOTATION_PROPS["sourceFile"], "本体/extracted_elderly_chronic.owl")

    bucket_to_parent = {
        "named_or_generic_pattern": CLASS_NAMES["NamedDietaryPattern"],
        "composition_or_restriction": CLASS_NAMES["RestrictionDietaryPattern"],
        "timing_pattern": CLASS_NAMES["TimingDietaryPattern"],
    }

    for row in reference_rows:
        iri = DIETA[f"{row['concept_id'].replace(':', '_')}_{slugify(row['canonical_label'])}"]
        g.add((iri, RDF.type, OWL.Class))
        g.add((iri, RDFS.subClassOf, bucket_to_parent.get(row["curation_bucket"], CLASS_NAMES["DietaryPattern"])))
        add_literal(g, iri, RDFS.label, row["canonical_label"])
        add_literal(g, iri, ANNOTATION_PROPS["preferredSource"], row["preferred_source"])
        add_literal(g, iri, ANNOTATION_PROPS["preferredSourceId"], row["preferred_source_id"])
        add_literal(g, iri, ANNOTATION_PROPS["preferredSourceUrl"], row["preferred_source_url"])
        add_literal(g, iri, ANNOTATION_PROPS["curationBucket"], row["curation_bucket"])
        add_literal(g, iri, ANNOTATION_PROPS["mappingStatus"], "reference_term")
        for alias in parse_json_field(row["all_aliases"], []):
            add_literal(g, iri, ANNOTATION_PROPS["aliasText"], alias)
        source_url = row["preferred_source_url"]
        if source_url and source_url.startswith("http"):
            g.add((iri, SKOS.exactMatch, URIRef(source_url)))

    for idx, row in enumerate(manual_rows, start=1):
        iri = DIETA[f"DPAT_CAND_{idx:03d}_{slugify(row['observed_term'])}"]
        g.add((iri, RDF.type, OWL.Class))
        g.add((iri, RDFS.subClassOf, CLASS_NAMES["DietaryPatternCandidate"]))
        add_literal(g, iri, RDFS.label, row["observed_term"])
        add_literal(g, iri, ANNOTATION_PROPS["mappingStatus"], row["mapping_status"])
        add_literal(g, iri, ANNOTATION_PROPS["curationBucket"], "manual_review_candidate")
        add_literal(g, iri, ANNOTATION_PROPS["schemaOriginPrompt"], "grounding_data/dietary_pattern/dietary_pattern_manual_review.csv")
        suggestions = parse_json_field(row["suggested_matches"], [])
        for suggestion in suggestions[:3]:
            label = suggestion.get("canonical_label", "")
            concept_id = suggestion.get("concept_id", "")
            similarity = suggestion.get("similarity", "")
            add_literal(g, iri, ANNOTATION_PROPS["suggestedMatch"], f"{label} [{concept_id}] similarity={similarity}")

    return g


def write_csv_json(rows: List[Dict[str, Any]], stem: str) -> None:
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    json_path = OUTPUT_DIR / f"{stem}.json"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        json_path.write_text("[]", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def save_graph(graph: Graph, stem: str) -> None:
    owl_path = OUTPUT_DIR / f"{stem}.owl"
    ttl_path = OUTPUT_DIR / f"{stem}.ttl"
    owl_path.write_bytes(graph.serialize(format="pretty-xml", encoding="utf-8"))
    ttl_path.write_text(graph.serialize(format="turtle"), encoding="utf-8")


def build_markdown_summary(
    entity_rows: List[Dict[str, Any]],
    relation_rows: List[Dict[str, Any]],
    topic_rows: List[Dict[str, Any]],
    reference_rows: List[Dict[str, Any]],
    manual_rows: List[Dict[str, Any]],
) -> str:
    lines = [
        "# DIETA 任务本体约束层与桥接本体摘要",
        "",
        "## 任务本体约束层",
        "",
        f"- 实体类型数：{len(entity_rows)}",
        f"- 正式关系数：7（另保留 1 个 prompt-surface relation 标签说明项）",
        f"- Topic clusters：{len(topic_rows)}",
        "",
        "### 实体类型",
        "",
        "| Entity | Router definition | Runtime grounding channel | Model support |",
        "|---|---|---|---|",
    ]
    for row in entity_rows:
        lines.append(
            f"| {row['entity_type']} | {row['router_definition']} | {row['runtime_grounding_channel']} | {row['model_supported_topics'] or 'prompt_only_or_none'} |"
        )
    lines.extend(
        [
            "",
            "### Topic clusters",
            "",
            "| Topic | Head | Relations | Tail | FD docs | FD grounded triples |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in topic_rows:
        lines.append(
            f"| {row['topic_id']} ({row['topic_name']}) | {row['head_type']} | {row['relation_space'].replace('|', ', ')} | {row['tail_type']} | {row['fd_document_count']} | {row['fd_grounded_triple_count']} |"
        )
    lines.extend(
        [
            "",
            "### DietaryPattern branch",
            "",
            f"- Curated reference concepts included in OWL: {len(reference_rows)}",
            f"- Manual-review candidate concepts preserved in OWL: {len(manual_rows)}",
            "",
            "## 论文可直接使用的结论",
            "",
            "DIETA 的任务本体约束层可以被显式整理为 4 类实体（Food、Nutrient、Disease、DietaryPattern）、7 类正式关系（recommended_for、avoid_for、beneficial、harmful、has_nutrient、includes_food、provides_nutrient）以及 6 个 topic clusters（T1–T6）。",
            "",
            "与运行时 grounding 资源不同，重建后的桥接 OWL 不声称替代当前 HumanDO/USDA FDC 的实际对齐通道，而是将 FoodOn、CDNO、ICD10CM chronic subset 与 MeSH/ONS dietary pattern lexicon 显式组织为可复用的语义桥接层，用于说明 schema 边界、保留外部语义锚点并支持后续本体扩展。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_readme(summary: Dict[str, Any]) -> str:
    lines = [
        "# DIETA Task Ontology Assets",
        "",
        "本目录保存基于 DIETA prompts、config、FD 主实验结果和本体资源重建的任务本体约束层与桥接本体工件。",
        "",
        "## 主要文件",
        "",
        "- `DIETA_task_constraint_layer.owl` / `DIETA_task_constraint_layer.ttl`",
        "  - 显式定义 4 类实体、7 类正式关系和 6 个 topic clusters。",
        "- `DIETA_semantic_bridge.owl` / `DIETA_semantic_bridge.ttl`",
        "  - 在任务本体约束层之上，接入 FoodOn、CDNO、ICD10CM chronic subset，以及 MeSH/ONS DietaryPattern 词表与 manual-review candidates。",
        "- `task_entity_types.*`",
        "  - 实体类型、router 定义、模型支持与 runtime grounding 通道。",
        "- `task_relations.*`",
        "  - 正式关系空间、合法 domain/range、schema 来源与 FD grounded triple 计数。",
        "- `topic_clusters.*`",
        "  - T1–T6 的 head/tail、relation space、模型支持、FD 文档覆盖与 grounded triple 计数。",
        "- `legal_relation_patterns.*`",
        "  - 合法 head-relation-tail 模式清单，可直接用于论文表格或规则检查。",
        "- `external_ontology_sources.*`",
        "  - 当前项目中与 Food、Nutrient、Disease、DietaryPattern 相关的外部资源来源与状态说明。",
        "- `task_ontology_for_manuscript.md`",
        "  - 面向论文写作的摘要版说明。",
        "- `ontology_build_summary.json`",
        "  - 生成统计与时间戳。",
        "",
        "## 设计说明",
        "",
        "1. 任务本体约束层（task constraint layer）",
        "   - 对应论文中的操作性 schema：限定实体类型、关系空间、topic clusters 和合法语义模式。",
        "2. 语义桥接层（semantic bridge）",
        "   - 对应论文中用于说明外部语义锚点和本体来源的 OWL 工件。",
        "   - 它不等同于当前运行时的全部 grounding 通道；当前 runtime 仍主要依赖 HumanDO、USDA FoodData Central 与 DietaryPattern 词表。",
        "3. 历史文件说明",
        "   - `本体/老年人膳食知识图谱.owl` 被识别为历史工件，解析时存在格式问题，因此未作为本次重建的主来源，但在 `external_ontology_sources.*` 中被保留说明。",
        "",
        "## 本次生成统计",
        "",
        f"- entity_types: {summary['entity_type_count']}",
        f"- formal_relations: {summary['formal_relation_count']}",
        f"- topic_clusters: {summary['topic_cluster_count']}",
        f"- dietary_pattern_reference_terms: {summary['dietary_pattern_reference_terms']}",
        f"- dietary_pattern_manual_review_candidates: {summary['dietary_pattern_manual_review_candidates']}",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topic_config = load_topic_config()
    router_defs = parse_router_entity_definitions()
    relation_defs = extract_relation_definitions(topic_config)
    seed_topic_counts = load_seed_topic_counts()
    grounding_topic_counts, grounding_relation_counts = load_grounding_counts()
    reference_rows, manual_rows = load_dietary_pattern_resources()

    entity_rows = build_entity_rows(topic_config, router_defs)
    relation_rows = build_relation_rows(topic_config, relation_defs, grounding_relation_counts)
    topic_rows = build_topic_rows(topic_config, seed_topic_counts, grounding_topic_counts)
    legal_pattern_rows = build_legal_pattern_rows(topic_rows)

    write_csv_json(entity_rows, "task_entity_types")
    write_csv_json(relation_rows, "task_relations")
    write_csv_json(topic_rows, "topic_clusters")
    write_csv_json(legal_pattern_rows, "legal_relation_patterns")
    write_csv_json(EXTERNAL_SOURCE_ROWS, "external_ontology_sources")

    task_graph = build_task_ontology_graph(entity_rows, relation_rows, topic_rows)
    bridge_graph = build_bridge_graph(task_graph, reference_rows, manual_rows)
    save_graph(task_graph, "DIETA_task_constraint_layer")
    save_graph(bridge_graph, "DIETA_semantic_bridge")

    manuscript_md = build_markdown_summary(entity_rows, relation_rows, topic_rows, reference_rows, manual_rows)
    (OUTPUT_DIR / "task_ontology_for_manuscript.md").write_text(manuscript_md, encoding="utf-8")

    summary = {
        "entity_type_count": len(entity_rows),
        "formal_relation_count": 7,
        "topic_cluster_count": len(topic_rows),
        "dietary_pattern_reference_terms": len(reference_rows),
        "dietary_pattern_manual_review_candidates": len(manual_rows),
        "seed_topic_document_counts": dict(seed_topic_counts),
        "grounding_topic_triple_counts": dict(grounding_topic_counts),
        "grounding_relation_triple_counts": dict(grounding_relation_counts),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "ontology_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(build_readme(summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



