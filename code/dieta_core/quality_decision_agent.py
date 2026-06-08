#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_decision.py - Critic Agent（评审智能体）：RUQA质量评估与反思

功能：
- 整合四智能体架构（Triage/Extraction/Critic/Grounding）
- 实现两条反馈回路机制
- RUQA质量评估方法
- 三级质量分流
- 反思报告生成
- 完整的端到端流程

按照DIETA框架技术说明文档v3.0实现
"""

import os
import time
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from settings import (
    API_KEY, API_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    API_RETRY_TIMES, API_RETRY_INTERVAL,
    NER_PROMPT_COUNT, RE_PROMPT_COUNT, TAU, DELTA,
    THETA_HIGH, THETA_LOW, LAMBDA_UNCERTAINTY,
    K_MAX_FEEDBACK, BETA_DECAY,
    SEMANTIC_TRIAGE_LATEST, SEED_ENTITY_LATEST,
    CANDIDATE_EXTRACTION_NER_LATEST_JSON, CANDIDATE_EXTRACTION_RE_LATEST_JSON,
    QUALITY_DECISION_DIR, QUALITY_DECISION_LATEST_XLSX, QUALITY_DECISION_LATEST_JSON,
    TOPIC_CONFIG, PROMPTS_DIR
)
from model_client import LLMClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityTier(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ReflectionReport:
    identified_issue: str
    root_cause: str
    remediation_suggestion: str


@dataclass
class RUQAScore:
    quality_score: float
    uncertainty_score: float
    final_score: float
    quality_tier: QualityTier
    p_eh: float
    p_et: float
    p_r: float
    lambda_reflect: float = 1.0
    quality_score_penalized: float = 0.0
    feedback_rounds: int = 0
    u_initial: float = 0.0  # U(0)(t): 第一轮计算的不确定性分数，固定不变


@dataclass
class ProcessResult:
    pmid: str
    triage_result: Optional[Dict] = None
    seed_entities: Optional[Dict] = None
    extraction_results: List[Dict] = None
    re_results: List[Dict] = None
    ruqa_scores: List[RUQAScore] = None
    reflection_reports: List[ReflectionReport] = None
    final_entities: List[Dict] = None
    final_triples: List[Dict] = None
    feedback_count: int = 0
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


class DIETACriticController:
    """Critic Agent（评审智能体）主控制器"""
    
    def __init__(self):
        """初始化Critic控制器"""
        self.all_results: Dict[str, ProcessResult] = {}
        os.makedirs(QUALITY_DECISION_DIR, exist_ok=True)
        
        self.llm_client = LLMClient()
        self._load_critic_prompt()
        
        logger.info("=" * 70)
        logger.info("Critic Agent（评审智能体）初始化完成")
        logger.info("=" * 70)
        self._print_params()
    
    def _load_critic_prompt(self):
        """加载Critic Agent的提示词模板和所有主题的NER/RE提示词"""
        self.critic_prompt = """
You are a professional medical knowledge graph quality review expert. Please review the following extraction results, identify potential issues, and generate a reflection report.

Original Text (Abstract):
{text}

Extraction Results:
{extraction_result}

RUQA Scoring Information:
- Head Entity Confidence P(eh): {p_eh:.3f}
- Tail Entity Confidence P(et): {p_et:.3f}
- Relation Confidence P(r): {p_r:.3f}
- Uncertainty Score U(t): {uncertainty:.3f}
- Final Score Score(t): {score:.3f}
- Quality Tier: {quality_tier}

Please output the review results in JSON format:
{{
    "reflection_report": {{
        "identified_issue": "Identified main issue (please describe in English, based on the above scores and extraction results)",
        "root_cause": "Root cause of the issue (please describe in English, deeply analyze possible reasons)",
        "remediation_suggestion": "Specific improvement suggestions (please describe in English, provide actionable recommendations)"
    }}
}}
"""
        
        # 加载所有主题的NER和RE提示词
        self.ner_prompts = {}
        self.re_prompts = {}
        
        for topic in TOPIC_CONFIG.keys():
            self.ner_prompts[topic] = {}
            self.re_prompts[topic] = {}
            
            # 加载NER提示词
            for i in range(1, NER_PROMPT_COUNT + 1):
                prompt_file = os.path.join(PROMPTS_DIR, f"ner_{topic}_p{i}.txt")
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        self.ner_prompts[topic][f"p{i}"] = f.read().strip()
            
            # 加载RE提示词
            for i in range(1, RE_PROMPT_COUNT + 1):
                prompt_file = os.path.join(PROMPTS_DIR, f"re_{topic}_q{i}.txt")
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        self.re_prompts[topic][f"q{i}"] = f.read().strip()
        
        logger.info(f"已加载提示词: {len(self.ner_prompts)}个主题NER, {len(self.re_prompts)}个主题RE")
    
    def _print_params(self):
        """打印参数摘要"""
        print("\n" + "=" * 70)
        print("Critic Agent (RUQA) 当前参数配置")
        print("=" * 70)
        print(f"Extraction Agent:")
        print(f"  NER提示词数: {NER_PROMPT_COUNT}, TAU: {TAU}")
        print(f"  RE提示词数: {RE_PROMPT_COUNT}, DELTA: {DELTA}")
        print()
        print("Critic Agent (RUQA):")
        print(f"  THETA_HIGH: {THETA_HIGH}, THETA_LOW: {THETA_LOW}")
        print(f"  LAMBDA_UNCERTAINTY: {LAMBDA_UNCERTAINTY}")
        print(f"  K_MAX_FEEDBACK: {K_MAX_FEEDBACK}, BETA_DECAY: {BETA_DECAY}")
        print()
        print("=" * 70 + "\n")
    
    def load_existing_data(self):
        """加载已有的处理结果
        
        Returns:
            tuple: (ner_data, re_data, seed_data)
        """
        ner_data = {}
        re_data = {}
        seed_data = {}
        
        if os.path.exists(SEED_ENTITY_LATEST):
            with open(SEED_ENTITY_LATEST, 'r', encoding='utf-8') as f:
                seed_data = json.load(f)
            logger.info(f"已加载种子实体数据: {len(seed_data)} 条")
        
        if os.path.exists(CANDIDATE_EXTRACTION_NER_LATEST_JSON):
            with open(CANDIDATE_EXTRACTION_NER_LATEST_JSON, 'r', encoding='utf-8') as f:
                ner_data = json.load(f)
            logger.info(f"已加载NER数据: {len(ner_data)} 条")
        
        if os.path.exists(CANDIDATE_EXTRACTION_RE_LATEST_JSON):
            with open(CANDIDATE_EXTRACTION_RE_LATEST_JSON, 'r', encoding='utf-8') as f:
                re_data = json.load(f)
            logger.info(f"已加载RE数据: {len(re_data)} 条")
        
        return ner_data, re_data, seed_data
    
    def calculate_ruqa_score(self, 
                            p_entity_head: float, 
                            p_entity_tail: float, 
                            p_relation: float,
                            raw_votes: Optional[Dict] = None,
                            feedback_round: int = 0,
                            u_initial: Optional[float] = None) -> RUQAScore:
        """计算RUQA评分
        
        Args:
            p_entity_head: 头实体置信度
            p_entity_tail: 尾实体置信度
            p_relation: 关系置信度
            raw_votes: 关系投票分布
            feedback_round: 反馈轮次
            u_initial: 第一轮计算的不确定性分数（固定不变）
            
        Returns:
            RUQAScore: RUQA评分结果
        """
        quality_score = p_entity_head * p_entity_tail * p_relation
        
        # 计算当前轮的不确定性分数
        current_uncertainty = self._calculate_uncertainty(raw_votes)
        
        # 如果是第一轮，保存u_initial
        if feedback_round == 0:
            u_initial = current_uncertainty
        elif u_initial is None:
            u_initial = current_uncertainty
        
        # 使用固定的U(0)(t)，不随迭代轮次改变
        uncertainty_score = u_initial
        
        lambda_reflect = 1.0 - BETA_DECAY * (feedback_round / K_MAX_FEEDBACK) if feedback_round > 0 else 1.0
        quality_score_penalized = lambda_reflect * quality_score
        
        # Score(k)(t) = λ_reflect(k) · Q(k)(t) - λ · U(0)(t)
        final_score = quality_score_penalized - LAMBDA_UNCERTAINTY * uncertainty_score
        
        quality_tier = self._determine_quality_tier(final_score)
        
        return RUQAScore(
            quality_score=quality_score,
            uncertainty_score=uncertainty_score,
            final_score=final_score,
            quality_tier=quality_tier,
            p_eh=p_entity_head,
            p_et=p_entity_tail,
            p_r=p_relation,
            lambda_reflect=lambda_reflect,
            quality_score_penalized=quality_score_penalized,
            feedback_rounds=feedback_round,
            u_initial=u_initial
        )
    
    def _calculate_uncertainty(self, raw_votes: Optional[Dict]) -> float:
        """计算归一化Shannon熵
        
        Args:
            raw_votes: 关系投票分布
            
        Returns:
            float: 不确定性分数 [0, 1]
        """
        if not raw_votes or len(raw_votes) == 0:
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
        
        max_entropy = math.log2(min(RE_PROMPT_COUNT, len(vote_counts) + 1)) if vote_counts else 1.0
        normalized_uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return min(normalized_uncertainty, 1.0)
    
    def _determine_quality_tier(self, final_score: float) -> QualityTier:
        """确定质量等级"""
        if final_score >= THETA_HIGH:
            return QualityTier.HIGH
        elif final_score >= THETA_LOW:
            return QualityTier.MEDIUM
        else:
            return QualityTier.LOW
    
    def generate_reflection_report(self, 
                                   ruqa_score: RUQAScore,
                                   entities: List[Dict],
                                   triples: List[Dict],
                                   abstract: str) -> ReflectionReport:
        """生成反思报告（使用真实LLM调用）
        
        Args:
            ruqa_score: RUQA评分
            entities: 实体列表
            triples: 三元组列表
            abstract: 原始摘要
            
        Returns:
            ReflectionReport: 反思报告
        """
        try:
            extraction_dict = {
                "entities": entities,
                "triples": triples
            }
            
            prompt = self.critic_prompt.format(
                text=abstract,
                extraction_result=json.dumps(extraction_dict, ensure_ascii=False, indent=2),
                p_eh=ruqa_score.p_eh,
                p_et=ruqa_score.p_et,
                p_r=ruqa_score.p_r,
                uncertainty=ruqa_score.uncertainty_score,
                score=ruqa_score.final_score,
                quality_tier=ruqa_score.quality_tier.value
            )
            
            response = self.llm_client.generate(
                prompt=prompt,
                model=LLM_MODEL,
                temperature=0.4,
                max_tokens=1500
            )
            
            critic_data = self._parse_critic_response(response)
            
            return ReflectionReport(
                identified_issue=critic_data.get("reflection_report", {}).get("identified_issue", ""),
                root_cause=critic_data.get("reflection_report", {}).get("root_cause", ""),
                remediation_suggestion=critic_data.get("reflection_report", {}).get("remediation_suggestion", "")
            )
            
        except Exception as e:
            logger.error(f"生成反思报告失败: {e}", exc_info=True)
            return ReflectionReport(
                identified_issue=f"审查失败: {str(e)}",
                root_cause="系统错误",
                remediation_suggestion="请重新尝试"
            )
    
    def _parse_critic_response(self, response: str) -> Dict:
        """解析Critic Agent的JSON响应"""
        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"解析审查响应失败，使用默认值: {str(e)}")
            return {
                "reflection_report": {
                    "identified_issue": "未能生成详细评估",
                    "root_cause": "响应解析失败",
                    "remediation_suggestion": "请重新审查"
                }
            }
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON字符串"""
        text = text.strip()
        
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace:last_brace+1]
            return json_str
        
        return text
    
    def _analyze_reflection_issue(self, reflection_report: ReflectionReport) -> str:
        """分析反思报告，判断需要重跑NER+RE还是仅RE
        
        Args:
            reflection_report: 反思报告
            
        Returns:
            str: "ner_re" 或 "re_only"
        """
        issue = reflection_report.identified_issue.lower()
        
        # 如果问题指向实体置信度不足，需要重跑NER+RE
        entity_keywords = [
            "entity confidence", "entity extraction", "head entity", "tail entity",
            "p(eh", "p(et", "entity recognition", "entity identification",
            "实体置信度", "实体识别", "头实体", "尾实体"
        ]
        
        for keyword in entity_keywords:
            if keyword in issue:
                return "ner_re"
        
        # 默认只重跑RE
        return "re_only"
    
    def _refine_triple_with_reflection(self, 
                                      triple: Dict, 
                                      reflection_report: ReflectionReport,
                                      abstract: str,
                                      pmid: str,
                                      seed_entities: Dict,
                                      current_entities: List[Dict]) -> Dict:
        """
        用反思报告改进单个三元组（调用候选抽取模块）
        
        根据反思报告的诊断结果选择性触发NER+RE或仅RE的重抽取：
        - 如果 Issue 指向实体置信度不足 → 重新执行 NER+RE
        - 如果 Issue 指向关系置信度不足且实体置信度已够 → 只重新执行 RE
        
        Args:
            triple: 原始三元组
            reflection_report: 反思报告
            abstract: 原始摘要
            pmid: 文献ID
            seed_entities: 种子实体
            current_entities: 当前实体列表
            
        Returns:
            Dict: 改进后的三元组（包含新的p_eh, p_et, p_r）
        """
        topic = triple.get('topic', 'T1')
        
        # 分析反思报告，判断需要重跑NER+RE还是仅RE
        refine_type = self._analyze_reflection_issue(reflection_report)
        
        logger.info(f"      反思诊断: {refine_type.upper()} (基于: {reflection_report.identified_issue[:50]}...)")
        
        # 构建反思报告文本
        reflection_text = f"""
[Reflection Report]
Issue: {reflection_report.identified_issue}
Root Cause: {reflection_report.root_cause}
Suggestion: {reflection_report.remediation_suggestion}

[Triple to Refine]
Subject: {triple.get('subject', '')}
Relation: {triple.get('relation', '')}
Object: {triple.get('object', '')}
"""
        
        try:
            if refine_type == "ner_re":
                # 重跑NER+RE
                logger.info(f"      重新执行 NER+RE...")
                return self._run_ner_and_re(pmid, abstract, seed_entities, topic, reflection_text, triple)
            else:
                # 只重跑RE
                logger.info(f"      只重新执行 RE（保持实体不变）...")
                return self._run_re_only(pmid, abstract, current_entities, topic, reflection_text, triple)
                
        except Exception as e:
            logger.error(f"      反馈重抽取失败: {e}")
            return triple
    
    def _run_ner_and_re(self, pmid: str, abstract: str, seed_entities: Dict, 
                        topic: str, reflection_text: str, original_triple: Dict) -> Dict:
        """重新执行NER+RE
        
        Args:
            pmid: 文献ID
            abstract: 摘要
            seed_entities: 种子实体
            topic: 主题
            reflection_text: 反思报告文本
            original_triple: 原始三元组
            
        Returns:
            Dict: 新的三元组（包含新的p_eh, p_et, p_r）
        """
        from candidate_entity_agent import run_ner_with_reflection
        from candidate_relation_agent import run_re_with_reflection
        
        # 1. 重新执行NER
        ner_result = run_ner_with_reflection(
            pmid=pmid,
            abstract=abstract,
            seed_entities=seed_entities,
            topic=topic,
            reflection_report=reflection_text
        )
        
        new_entities = ner_result.get('entities', [])
        
        if not new_entities:
            logger.warning(f"      NER重抽取未返回实体，返回原始三元组")
            return original_triple
        
        # 2. 重新执行RE
        re_result = run_re_with_reflection(
            pmid=pmid,
            abstract=abstract,
            entities=new_entities,
            topic=topic,
            reflection_report=reflection_text
        )
        
        new_triples = re_result.get('triples', [])
        
        if not new_triples:
            logger.warning(f"      RE重抽取未返回三元组，返回原始三元组")
            return original_triple
        
        # 3. 找到与原始三元组最匹配的新三元组
        best_triple = self._find_best_matching_triple(new_triples, original_triple)
        
        if best_triple:
            # 计算新的置信度
            new_p_eh = best_triple.get('subject_confidence', 0.8)
            new_p_et = best_triple.get('object_confidence', 0.8)
            new_p_r = best_triple.get('confidence', 0.8)
            
            result = best_triple.copy()
            result['p_eh'] = new_p_eh
            result['p_et'] = new_p_et
            result['p_relation'] = new_p_r
            result['refine_type'] = 'ner_re'
            
            logger.info(f"      NER+RE重抽取成功: "
                       f"subject={best_triple.get('subject')}({new_p_eh:.2f}), "
                       f"relation={best_triple.get('relation')}({new_p_r:.2f}), "
                       f"object={best_triple.get('object')}({new_p_et:.2f})")
            
            return result
        
        return original_triple
    
    def _run_re_only(self, pmid: str, abstract: str, entities: List[Dict], 
                     topic: str, reflection_text: str, original_triple: Dict) -> Dict:
        """只重新执行RE（保持实体不变）
        
        Args:
            pmid: 文献ID
            abstract: 摘要
            entities: 当前实体列表
            topic: 主题
            reflection_text: 反思报告文本
            original_triple: 原始三元组
            
        Returns:
            Dict: 新的三元组（p_eh和p_et保持不变，只更新p_r）
        """
        from candidate_relation_agent import run_re_with_reflection
        
        # 只重新执行RE
        re_result = run_re_with_reflection(
            pmid=pmid,
            abstract=abstract,
            entities=entities,
            topic=topic,
            reflection_report=reflection_text
        )
        
        new_triples = re_result.get('triples', [])
        
        if not new_triples:
            logger.warning(f"      RE重抽取未返回三元组，返回原始三元组")
            return original_triple
        
        # 找到与原始三元组最匹配的新三元组
        best_triple = self._find_best_matching_triple(new_triples, original_triple)
        
        if best_triple:
            # 只更新p_r，保持p_eh和p_et不变
            new_p_r = best_triple.get('confidence', 0.8)
            original_p_eh = original_triple.get('p_eh', 0.8)
            original_p_et = original_triple.get('p_et', 0.8)
            
            result = best_triple.copy()
            result['p_eh'] = original_p_eh  # 保持原始实体置信度
            result['p_et'] = original_p_et  # 保持原始实体置信度
            result['p_relation'] = new_p_r
            result['refine_type'] = 're_only'
            
            logger.info(f"      RE重抽取成功: "
                       f"relation={best_triple.get('relation')}({new_p_r:.2f}), "
                       f"实体置信度保持不变: p_eh={original_p_eh:.2f}, p_et={original_p_et:.2f}")
            
            return result
        
        return original_triple
    
    def _find_best_matching_triple(self, new_triples: List[Dict], original_triple: Dict) -> Optional[Dict]:
        """找到与原始三元组最匹配的新三元组
        
        Args:
            new_triples: 新的三元组列表
            original_triple: 原始三元组
            
        Returns:
            Optional[Dict]: 最匹配的三元组
        """
        if not new_triples:
            return None
        
        original_subject = original_triple.get('subject', '').lower()
        original_object = original_triple.get('object', '').lower()
        
        best_match = None
        best_score = 0
        
        for triple in new_triples:
            new_subject = triple.get('subject', '').lower()
            new_object = triple.get('object', '').lower()
            
            # 计算匹配分数
            score = 0
            if original_subject in new_subject or new_subject in original_subject:
                score += 1
            if original_object in new_object or new_object in original_object:
                score += 1
            
            # 优先选择置信度高的
            score += triple.get('confidence', 0) * 0.1
            
            if score > best_score:
                best_score = score
                best_match = triple
        
        return best_match
    
    def _parse_refinement_response(self, response: str) -> Optional[Dict]:
        """解析改进抽取的响应"""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            # 尝试找到三元组
            if 'triples' in data and len(data['triples']) > 0:
                return data['triples'][0]
            elif 'triple' in data:
                return data['triple']
            elif 'subject' in data and 'relation' in data and 'object' in data:
                return data
        except Exception as e:
            logger.warning(f"解析改进抽取响应失败: {e}")
        return None
    
    def _vote_on_refined_results(self, all_results: List[Dict], original_triple: Dict) -> Dict:
        """
        对改进抽取结果进行自一致性投票
        
        Args:
            all_results: 所有提示词的改进抽取结果
            original_triple: 原始三元组
            
        Returns:
            Dict: 投票后的三元组
        """
        if not all_results:
            return original_triple
        
        # 统计关系
        relation_counts = {}
        subject_counts = {}
        object_counts = {}
        
        for item in all_results:
            result = item['result']
            rel = result.get('relation', '')
            subj = result.get('subject', '')
            obj = result.get('object', '')
            
            if rel:
                relation_counts[rel] = relation_counts.get(rel, 0) + 1
            if subj:
                subject_counts[subj] = subject_counts.get(subj, 0) + 1
            if obj:
                object_counts[obj] = object_counts.get(obj, 0) + 1
        
        # 选择投票最多的
        def get_best(counts: Dict, default: str) -> str:
            if not counts:
                return default
            return max(counts.items(), key=lambda x: x[1])[0]
        
        best_subject = get_best(subject_counts, original_triple.get('subject', ''))
        best_relation = get_best(relation_counts, original_triple.get('relation', ''))
        best_object = get_best(object_counts, original_triple.get('object', ''))
        
        # 计算新的置信度
        total_votes = len(all_results)
        
        def get_confidence(counts: Dict, best_val: str) -> float:
            if best_val not in counts:
                return 0.8
            return counts[best_val] / total_votes
        
        new_p_eh = get_confidence(subject_counts, best_subject)
        new_p_r = get_confidence(relation_counts, best_relation)
        new_p_et = get_confidence(object_counts, best_object)
        
        # 返回改进后的三元组
        new_triple = original_triple.copy()
        new_triple['subject'] = best_subject
        new_triple['relation'] = best_relation
        new_triple['object'] = best_object
        new_triple['p_eh'] = new_p_eh
        new_triple['p_et'] = new_p_et
        new_triple['p_relation'] = new_p_r
        new_triple['confidence'] = new_p_r
        
        logger.info(f"改进投票结果: "
                   f"subject={best_subject}({new_p_eh:.2f}), "
                   f"relation={best_relation}({new_p_r:.2f}), "
                   f"object={best_object}({new_p_et:.2f})")
        
        return new_triple
    
    def process_single_pmid(self, 
                          pmid: str,
                          ner_data: Dict,
                          re_data: Dict,
                          seed_data: Dict) -> ProcessResult:
        """处理单个PMID的完整流程
        
        Args:
            pmid: PMID
            ner_data: NER数据
            re_data: RE数据
            seed_data: 种子实体数据
            
        Returns:
            ProcessResult: 处理结果
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info(f"处理 PMID: {pmid}")
        logger.info("=" * 60)
        
        result = ProcessResult(
            pmid=pmid,
            extraction_results=[],
            re_results=[],
            ruqa_scores=[],
            reflection_reports=[],
            final_entities=[],
            final_triples=[]
        )
        
        try:
            ner_item = ner_data.get(pmid, {})
            re_item = re_data.get(pmid, {})
            seed_item = seed_data.get(pmid, {})
            
            result.seed_entities = seed_item.get('seed_entities', {})
            abstract = ner_item.get('abstract', re_item.get('abstract', ''))
            
            all_triples = []
            all_entities = []
            
            if 'entities' in ner_item:
                all_entities = ner_item['entities']
            
            if 're_results' in re_item:
                for topic, topic_result in re_item['re_results'].items():
                    if 'triples' in topic_result:
                        for triple in topic_result['triples']:
                            triple['topic'] = topic
                            triple['p_eh'] = triple.get('subject_confidence', triple.get('confidence', 0.8))
                            triple['p_et'] = triple.get('object_confidence', triple.get('confidence', 0.8))
                            triple['p_relation'] = triple.get('confidence', 0.8)
                            triple['subject_type'] = triple.get('subject_type', None)
                            triple['object_type'] = triple.get('object_type', None)
                            all_triples.append(triple)
            
            if not all_triples:
                logger.warning(f"PMID {pmid}: 未找到三元组")
                result.final_entities = all_entities
                result.final_triples = []
                result.processing_time = time.time() - start_time
                return result
            
            final_triples = []
            
            for triple_idx, triple in enumerate(all_triples):
                logger.info(f"  处理三元组 {triple_idx + 1}/{len(all_triples)}")
                
                p_eh = triple.get('p_eh', triple.get('confidence', 0.8))
                p_et = triple.get('p_et', triple.get('confidence', 0.8))
                p_r = triple.get('p_relation', triple.get('confidence', 0.8))
                raw_votes = triple.get('raw_votes', {})
                
                # 确保所有值都是浮点数，避免None
                p_eh = float(p_eh) if p_eh is not None else 0.8
                p_et = float(p_et) if p_et is not None else 0.8
                p_r = float(p_r) if p_r is not None else 0.8
                
                current_triple = triple.copy()
                current_ruqa_score = None
                current_reflection = None
                u_initial = None  # 第一轮计算的不确定性分数
                
                for feedback_round in range(K_MAX_FEEDBACK):
                    # 计算RUQA评分，传递u_initial
                    ruqa_score = self.calculate_ruqa_score(
                        p_eh, p_et, p_r, raw_votes, feedback_round, u_initial
                    )
                    
                    # 保存第一轮的u_initial
                    if feedback_round == 0:
                        u_initial = ruqa_score.u_initial
                    
                    logger.info(f"    迭代 {feedback_round + 1}: "
                              f"Q(t)={ruqa_score.quality_score:.3f}, "
                              f"U(0)(t)={ruqa_score.u_initial:.3f}, "
                              f"Score(t)={ruqa_score.final_score:.3f}, "
                              f"Tier={ruqa_score.quality_tier.value}")
                    
                    current_ruqa_score = ruqa_score
                    
                    if ruqa_score.quality_tier == QualityTier.HIGH:
                        logger.info(f"    ✓ 达到高质量，直接通过")
                        break
                    elif ruqa_score.quality_tier == QualityTier.MEDIUM:
                        if feedback_round < K_MAX_FEEDBACK - 1:
                            logger.info(f"    ⚠ 中等质量，生成反思报告并改进抽取")
                            reflection = self.generate_reflection_report(
                                ruqa_score, all_entities, [triple], abstract
                            )
                            current_reflection = reflection
                            result.reflection_reports.append(reflection)
                            result.feedback_count += 1
                            
                            logger.info(f"      Issue: {reflection.identified_issue}")
                            logger.info(f"      Root Cause: {reflection.root_cause}")
                            logger.info(f"      Suggestion: {reflection.remediation_suggestion}")
                            
                            # 用反思报告改进抽取（调用候选抽取模块）
                            logger.info(f"      用反思报告重新抽取...")
                            improved_triple = self._refine_triple_with_reflection(
                                current_triple, reflection, abstract, pmid,
                                result.seed_entities, all_entities
                            )
                            
                            # 更新三元组和置信度
                            current_triple = improved_triple
                            p_eh = improved_triple.get('p_eh', p_eh)
                            p_et = improved_triple.get('p_et', p_et)
                            p_r = improved_triple.get('p_relation', p_r)
                            
                            # 确保值都是浮点数
                            p_eh = float(p_eh) if p_eh is not None else 0.8
                            p_et = float(p_et) if p_et is not None else 0.8
                            p_r = float(p_r) if p_r is not None else 0.8
                            
                            # 更新raw_votes（如果有的话）
                            raw_votes = improved_triple.get('raw_votes', raw_votes)
                        else:
                            logger.info(f"    ⚠ 达到最大反馈次数，继续")
                            break
                    else:
                        logger.warning(f"    ✗ 低质量，丢弃")
                        # 低质量，不保留
                        current_triple = None
                        break
                
                if current_ruqa_score:
                    result.ruqa_scores.append(current_ruqa_score)
                
                # 只有不是None的三元组才保留
                if current_triple is not None:
                    triple_with_score = current_triple.copy()
                    if current_ruqa_score:
                        triple_with_score.update({
                            'Q_t': current_ruqa_score.quality_score,
                            'U_t': current_ruqa_score.uncertainty_score,
                            'Score_t': current_ruqa_score.final_score,
                            'quality_tier': current_ruqa_score.quality_tier.value,
                            'lambda_reflect': current_ruqa_score.lambda_reflect,
                            'Q_t_penalized': current_ruqa_score.quality_score_penalized,
                            'feedback_rounds': current_ruqa_score.feedback_rounds
                        })
                    
                    final_triples.append(triple_with_score)
                else:
                    logger.info(f"    三元组被丢弃（低质量）")
            
            result.final_entities = all_entities
            result.final_triples = final_triples
            result.success = True
            
            high_count = sum(1 for t in final_triples if t.get('quality_tier') == 'high')
            medium_count = sum(1 for t in final_triples if t.get('quality_tier') == 'medium')
            low_count = sum(1 for t in final_triples if t.get('quality_tier') == 'low')
            
            logger.info(f"PMID {pmid} 完成: "
                       f"实体={len(result.final_entities)}, "
                       f"三元组={len(result.final_triples)}, "
                       f"反馈次数={result.feedback_count}")
            logger.info(f"  质量分布: HIGH={high_count}, MEDIUM={medium_count}, LOW={low_count}")
            
        except Exception as e:
            logger.error(f"PMID {pmid} 处理失败: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
        
        result.processing_time = time.time() - start_time
        return result
    
    def process_all(self, save_interval: int = 5, max_records: int = None):
        """处理所有数据
        
        Args:
            save_interval: 每处理多少个PMID保存一次
            max_records: 最大处理记录数（用于批量运行控制）
        """
        logger.info("开始Critic Agent处理")
        
        ner_data, re_data, seed_data = self.load_existing_data()
        
        all_pmids = sorted(set(ner_data.keys()) | set(re_data.keys()))
        
        processed_ids = set()
        if os.path.exists(QUALITY_DECISION_LATEST_JSON):
            try:
                with open(QUALITY_DECISION_LATEST_JSON, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                processed_ids = set(existing.keys())
                logger.info(f"已加载 {len(processed_ids)} 条已处理记录")
            except Exception as e:
                logger.warning(f"读取已处理记录失败: {e}")
        
        pmids_to_process = [pmid for pmid in all_pmids if pmid not in processed_ids]
        
        # 限制处理记录数量
        if max_records:
            pmids_to_process = pmids_to_process[:max_records]
            logger.info(f"限制处理前 {max_records} 条记录")
        
        logger.info(f"共需处理 {len(all_pmids)} 个PMID，待处理 {len(pmids_to_process)} 个")
        
        all_results_dict = {}
        if processed_ids and os.path.exists(QUALITY_DECISION_LATEST_JSON):
            try:
                with open(QUALITY_DECISION_LATEST_JSON, 'r', encoding='utf-8') as f:
                    all_results_dict = json.load(f)
            except Exception as e:
                logger.warning(f"加载已处理结果失败: {e}")
        
        processed_count = 0
        for pmid in pmids_to_process:
            result = self.process_single_pmid(pmid, ner_data, re_data, seed_data)
            all_results_dict[pmid] = result
            self.all_results[pmid] = result
            processed_count += 1
            
            if processed_count % save_interval == 0:
                logger.info(f"已处理 {processed_count}/{len(pmids_to_process)}，保存进度...")
                self.save_results(all_results_dict)
        
        self.save_results(all_results_dict)
        
        return all_results_dict
    
    def save_results(self, results_dict: Dict[str, ProcessResult]):
        """保存结果"""
        results_json = {}
        rows_for_excel = []
        
        total_triples = 0
        high_count = 0
        medium_count = 0
        low_count = 0
        total_feedback = 0
        
        for pmid, result in results_dict.items():
            # 处理字典格式（从JSON加载的）和ProcessResult对象
            if isinstance(result, dict):
                result_dict = result
                final_entities = result.get('final_entities', [])
                final_triples = result.get('final_triples', [])
                feedback_count = result.get('feedback_count', 0)
                processing_time = result.get('processing_time', 0)
                success = result.get('success', True)
                error = result.get('error', None)
                reflection_reports = result.get('reflection_reports', [])
            else:
                result_dict = {
                    'pmid': pmid,
                    'final_entities': result.final_entities,
                    'final_triples': result.final_triples,
                    'feedback_count': result.feedback_count,
                    'processing_time': result.processing_time,
                    'success': result.success,
                    'error': result.error,
                    'reflection_reports': [asdict(report) for report in result.reflection_reports] if result.reflection_reports else []
                }
                final_entities = result.final_entities
                final_triples = result.final_triples
                feedback_count = result.feedback_count
                processing_time = result.processing_time
                success = result.success
                error = result.error
                reflection_reports = result_dict.get('reflection_reports', [])
            
            results_json[pmid] = result_dict
            
            for triple in final_triples:
                row = {
                    'pmid': pmid,
                    'subject': triple.get('subject', triple.get('head_entity', '')),
                    'subject_type': triple.get('subject_type', ''),
                    'relation': triple.get('relation', ''),
                    'object': triple.get('object', triple.get('tail_entity', '')),
                    'object_type': triple.get('object_type', ''),
                    'P_entity_head': triple.get('p_eh', triple.get('confidence', 0.0)),
                    'P_entity_tail': triple.get('p_et', triple.get('confidence', 0.0)),
                    'P_relation': triple.get('p_relation', triple.get('confidence', 0.0)),
                    'Q_t': triple.get('Q_t', 0.0),
                    'U_t': triple.get('U_t', 0.0),
                    'Score_t': triple.get('Score_t', 0.0),
                    'quality_tier': triple.get('quality_tier', 'unknown'),
                    'lambda_reflect': triple.get('lambda_reflect', 1.0),
                    'Q_t_penalized': triple.get('Q_t_penalized', 0.0),
                    'feedback_rounds': triple.get('feedback_rounds', 0),
                    'source_cluster': triple.get('source_cluster', triple.get('topic', ''))
                }
                rows_for_excel.append(row)
                
                total_triples += 1
                if row['quality_tier'] == 'high':
                    high_count += 1
                elif row['quality_tier'] == 'medium':
                    medium_count += 1
                else:
                    low_count += 1
            
            total_feedback += feedback_count
        
        with open(QUALITY_DECISION_LATEST_JSON, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_json = os.path.join(QUALITY_DECISION_DIR, f"quality_decision_{timestamp}.json")
        with open(archive_json, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)
        
        if rows_for_excel:
            import pandas as pd
            df = pd.DataFrame(rows_for_excel)
            df.to_excel(QUALITY_DECISION_LATEST_XLSX, index=False)
            
            archive_xlsx = os.path.join(QUALITY_DECISION_DIR, f"quality_decision_{timestamp}.xlsx")
            df.to_excel(archive_xlsx, index=False)
            
            logger.info(f"已保存 {len(rows_for_excel)} 个三元组到 {QUALITY_DECISION_LATEST_XLSX} 和 {archive_xlsx}")
        
        logger.info("=" * 60)
        logger.info("Critic Agent处理统计")
        logger.info("=" * 60)
        logger.info(f"总PMID数: {len(results_dict)}")
        logger.info(f"总三元组数量: {total_triples}")
        logger.info(f"  - HIGH质量: {high_count} ({high_count/total_triples*100:.1f}%)" if total_triples else 0)
        logger.info(f"  - MEDIUM质量: {medium_count} ({medium_count/total_triples*100:.1f}%)" if total_triples else 0)
        logger.info(f"  - LOW质量: {low_count} ({low_count/total_triples*100:.1f}%)" if total_triples else 0)
        logger.info(f"总反馈次数: {total_feedback}")
        logger.info(f"平均反馈/PMID: {total_feedback/len(results_dict):.2f}" if results_dict else 0)
        logger.info("=" * 60)
        
        logger.info(f"结果已保存:")
        logger.info(f"  - JSON: {QUALITY_DECISION_LATEST_JSON}")
        logger.info(f"  - Excel: {QUALITY_DECISION_LATEST_XLSX}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Critic Agent: 评审智能体')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量（用于批量运行控制）')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Critic Agent（评审智能体）启动")
    logger.info("四智能体架构 + 双反馈回路 + RUQA质量评估")
    if args.limit:
        logger.info(f"限制处理数量: {args.limit}")
    logger.info("=" * 70)
    
    controller = DIETACriticController()
    results = controller.process_all(max_records=args.limit)
    
    logger.info("=" * 70)
    logger.info("Critic Agent处理完成")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()



