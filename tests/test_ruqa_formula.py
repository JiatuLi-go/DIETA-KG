from dieta_agentic.ruqa_formula import calculate_ruqa_retention_score

def test_ruqa_score_basic():
    raw_votes = {'q1': 'beneficial', 'q2': 'beneficial', 'q3': 'harmful'}
    r = calculate_ruqa_retention_score(0.8, 0.9, 0.7, raw_votes, feedback_round=0, u_initial=None, re_prompt_count=5, lambda_uncertainty=0.3, beta_decay=0.5, k_max_feedback=3)
    assert 0 <= r.uncertainty_score <= 1
    assert abs(r.quality_score - 0.504) < 1e-9
    assert r.final_score <= r.quality_score
