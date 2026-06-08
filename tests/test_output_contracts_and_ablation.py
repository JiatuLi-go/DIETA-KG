import json
from pathlib import Path

from dieta_agentic.ablation import AblationConfig
from dieta_agentic.output_contracts import grounding_rows_from_json, quality_rows_from_json, GROUNDING_FINAL_COLUMNS, QUALITY_DECISION_COLUMNS


def test_no_feedback_variant_exports_runtime_flags():
    cfg = AblationConfig.from_variant('no_feedback')
    env = cfg.to_env()
    assert env['GCDI_ABLATION_DISABLE_FEEDBACK'] == '1'
    assert env['GCDI_ABLATION_DISABLE_GROUNDING_FEEDBACK'] == '1'


def test_after_feedback_json_exports_completed_rows(tmp_path: Path):
    path = tmp_path / 'grounding_alignment_latest.json'
    path.write_text(json.dumps({
        'feedback_completed': True,
        'aligned_triples': [{
            'pmid': '1',
            'subject': 'food',
            'relation': 'recommended_for',
            'object': 'disease',
            'source_cluster': 'T1',
            'quality_tier': 'high',
            'Score_t': 1.0,
            'needs_feedback': True,
            'head_alignment': {'status': 'direct_hit', 'standard_name': 'FOOD', 'similarity': 1.0},
            'tail_alignment': {'status': 'direct_hit', 'standard_name': 'disease', 'similarity': 1.0},
        }]
    }), encoding='utf-8')
    rows = grounding_rows_from_json(path)
    assert list(rows[0].keys()) == GROUNDING_FINAL_COLUMNS
    assert rows[0]['feedback_completed'] is True


def test_quality_json_exports_original_step4_column_order(tmp_path: Path):
    path = tmp_path / 'quality_decision_latest.json'
    path.write_text(json.dumps({
        '123': {
            'final_triples': [{
                'subject': 'whole grain',
                'subject_type': 'Food',
                'relation': 'recommended_for',
                'object': 'type 2 diabetes',
                'object_type': 'Disease',
                'p_eh': 0.8,
                'p_et': 0.9,
                'p_relation': 1.0,
                'Q_t': 0.72,
                'U_t': 0.0,
                'Score_t': 0.72,
                'quality_tier': 'high',
                'lambda_reflect': 1.0,
                'Q_t_penalized': 0.72,
                'feedback_rounds': 0,
                'source_cluster': 'T1',
            }]
        }
    }), encoding='utf-8')
    rows = quality_rows_from_json(path)
    assert list(rows[0].keys()) == QUALITY_DECISION_COLUMNS
    assert QUALITY_DECISION_COLUMNS[:6] == ['pmid', 'subject', 'subject_type', 'relation', 'object', 'object_type']
