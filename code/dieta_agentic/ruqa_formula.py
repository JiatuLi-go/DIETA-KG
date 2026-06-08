from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math

@dataclass(frozen=True)
class RUQAFormulaResult:
    quality_score: float
    uncertainty_score: float
    lambda_reflect: float
    quality_score_penalized: float
    final_score: float


def normalized_relation_entropy(raw_votes: Optional[Dict], re_prompt_count: int) -> float:
    """Normalized Shannon entropy used as the initial relation-uncertainty signal."""
    if not raw_votes:
        return 0.0
    vote_counts = {}
    for relation in raw_votes.values():
        if relation:
            vote_counts[relation] = vote_counts.get(relation, 0) + 1
    total_votes = sum(vote_counts.values())
    if total_votes == 0:
        return 0.0
    entropy = 0.0
    for count in vote_counts.values():
        p = count / total_votes
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(min(re_prompt_count, len(vote_counts) + 1)) if vote_counts else 1.0
    return min(entropy / max_entropy if max_entropy > 0 else 0.0, 1.0)


def calculate_ruqa_retention_score(
    p_entity_head: float,
    p_entity_tail: float,
    p_relation: float,
    raw_votes: Optional[Dict],
    *,
    feedback_round: int,
    u_initial: Optional[float],
    re_prompt_count: int,
    lambda_uncertainty: float,
    beta_decay: float,
    k_max_feedback: int,
) -> RUQAFormulaResult:
    """Paper-aligned DIETA retention score.

    Q(t) = p(e_h) * p(e_t) * p(r)
    U(0)(t) = normalized relation-distribution entropy in the first round
    lambda_reflect(k) = 1 - beta_decay * (k / K_max) for k > 0 else 1
    s^(k)(t) = lambda_reflect(k) * Q^(k)(t) - lambda * U(0)(t)
    """
    quality_score = p_entity_head * p_entity_tail * p_relation
    current_u = normalized_relation_entropy(raw_votes, re_prompt_count)
    if feedback_round == 0 or u_initial is None:
        u0 = current_u
    else:
        u0 = u_initial
    lambda_reflect = 1.0 - beta_decay * (feedback_round / k_max_feedback) if feedback_round > 0 else 1.0
    quality_score_penalized = lambda_reflect * quality_score
    final_score = quality_score_penalized - lambda_uncertainty * u0
    return RUQAFormulaResult(
        quality_score=quality_score,
        uncertainty_score=u0,
        lambda_reflect=lambda_reflect,
        quality_score_penalized=quality_score_penalized,
        final_score=final_score,
    )
