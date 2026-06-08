#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feedback_refinement.py - Feedback Loop ② Controller

Features:
- Read feedback_refinement_requests.json
- Process feedback requests: Call Extraction Agent to re-extract → Critic Agent to re-review → Grounding Agent to re-align
- Merge results to final output

Input:
- results/grounding_alignment/feedback_refinement_requests.json

Output:
- Updated results/grounding_alignment/grounding_alignment_latest.json
- Updated results/grounding_alignment/grounding_alignment_latest.xlsx
"""

import os
import sys
import json
import time
import logging
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from settings import (
    FEEDBACK_REFINEMENT_REQUESTS,
    GROUNDING_ALIGNMENT_LATEST_JSON,
    GROUNDING_ALIGNMENT_LATEST,
    CANDIDATE_EXTRACTION_NER_LATEST_JSON,
    CANDIDATE_EXTRACTION_RE_LATEST_JSON,
    QUALITY_DECISION_LATEST_XLSX,
    QUALITY_DECISION_LATEST_JSON,
    GROUNDING_ALIGNMENT_DIR
)
from utils import setup_logging

# Configure logging
logger = setup_logging(GROUNDING_ALIGNMENT_DIR, "feedback_refinement")


@dataclass
class FeedbackRequest:
    """Feedback Request Data Class"""
    pmid: str
    triple_idx: int
    original_triple: Dict
    feedback_entity: str
    feedback_entity_text: str
    candidate_name: str
    abstract: str
    is_retry: bool = False


class FeedbackLoopController:
    """Feedback Loop ② Controller"""
    
    def __init__(self):
        """Initialize Feedback Loop Controller"""
        self.feedback_requests: List[FeedbackRequest] = []
        self.original_results: Dict = {}
        self.updated_results: Dict = {}
        self._grounding_agent = None
        
        logger.info("=" * 70)
        logger.info("Feedback Loop ② Controller Initialized")
        logger.info("=" * 70)

    def _get_grounding_agent(self):
        """Initialize GroundingAgent once and reuse it for all feedback cases."""
        if self._grounding_agent is None:
            from grounding_alignment_agent import GroundingAgent

            logger.info("Initializing cached GroundingAgent for feedback re-alignment...")
            self._grounding_agent = GroundingAgent()
        return self._grounding_agent
    
    def load_feedback_requests(self) -> bool:
        """Load feedback requests
        
        Returns:
            bool: Whether there are feedback requests to process
        """
        if not os.path.exists(FEEDBACK_REFINEMENT_REQUESTS):
            logger.info("Feedback request file not found, no need to process Feedback Loop ②")
            return False
        
        try:
            with open(FEEDBACK_REFINEMENT_REQUESTS, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.feedback_requests = []
            for req in data.get('feedback_requests', []):
                req_copy = req.copy()
                # 兼容两种格式：'triple' 或 'original_triple'
                triple_data = req_copy.get('triple', req_copy.get('original_triple', {}))
                if 'triple' in req_copy and 'original_triple' not in req_copy:
                    req_copy['original_triple'] = req_copy.pop('triple')
                # 从triple中提取缺失的字段
                if 'feedback_entity' not in req_copy and 'feedback_entity' in triple_data:
                    req_copy['feedback_entity'] = triple_data['feedback_entity']
                if 'feedback_entity_text' not in req_copy and 'feedback_entity_text' in triple_data:
                    req_copy['feedback_entity_text'] = triple_data['feedback_entity_text']
                # 从triple的对齐信息中提取candidate_name
                if 'candidate_name' not in req_copy:
                    feedback_entity = req_copy.get('feedback_entity', 'head')
                    if feedback_entity == 'head' and 'head_alignment' in triple_data:
                        req_copy['candidate_name'] = triple_data['head_alignment'].get('standard_name', '')
                    elif feedback_entity == 'tail' and 'tail_alignment' in triple_data:
                        req_copy['candidate_name'] = triple_data['tail_alignment'].get('standard_name', '')
                self.feedback_requests.append(FeedbackRequest(**req_copy))
            
            logger.info(f"Loaded {len(self.feedback_requests)} feedback requests")
            return len(self.feedback_requests) > 0
            
        except Exception as e:
            logger.error(f"Failed to load feedback requests: {e}", exc_info=True)
            return False
    
    def load_original_results(self):
        """Load original alignment results"""
        if os.path.exists(GROUNDING_ALIGNMENT_LATEST_JSON):
            with open(GROUNDING_ALIGNMENT_LATEST_JSON, 'r', encoding='utf-8') as f:
                self.original_results = json.load(f)
            logger.info("Loaded original alignment results")
    
    def generate_feedback_report(self, request: FeedbackRequest) -> str:
        """Generate entity boundary ambiguity report
        
        Args:
            request: Feedback request
            
        Returns:
            str: Feedback report
        """
        report = f"""
[Entity Boundary Ambiguity Report]
PMID: {request.pmid}
Triple Index: {request.triple_idx}

Problem Description:
- Extracted Entity: {request.feedback_entity_text}
- Candidate Standard Name: {request.candidate_name}
- Issue: Extracted entity may have boundary truncation (extracted entity text is contained within candidate standard name)

Suggestion:
Please re-identify this entity within a broader context window to ensure the complete entity name is extracted.

Original Abstract Context:
{request.abstract[:2000]}
"""
        return report
    
    def call_extraction_agent_with_feedback(self, request: FeedbackRequest) -> Optional[Dict]:
        """Call Extraction Agent for feedback re-extraction
        
        反馈回路②：生成实体边界歧义报告，传回Extraction Agent请求在更宽泛的上下文窗口内重新执行实体识别。
        重新识别后再次进入对齐流程（is_retry标记为True），若仍未对齐则按开放世界验证流程处理，不再触发反馈回路②，防止死循环。
        
        Args:
            request: 反馈请求
            
        Returns:
            Dict: 重抽取结果
        """
        from candidate_entity_agent import run_ner_with_reflection
        from candidate_relation_agent import run_re_with_reflection
        
        logger.info("=" * 60)
        logger.info("Feedback Loop ②: Calling Extraction Agent for re-extraction")
        logger.info("=" * 60)
        
        # 生成实体边界歧义报告
        report = self.generate_feedback_report(request)
        logger.info(f"\nFeedback Report:\n{report}")
        
        try:
            # 1. 重新执行NER（在更宽泛的上下文窗口内）
            logger.info(f"Step 1: Re-running NER with broader context window...")
            
            # 从原始三元组获取信息
            original_triple = request.original_triple
            topic = original_triple.get('topic', original_triple.get('source_cluster', 'T1'))
            
            # 构建种子实体（从原始结果中获取）
            seed_entities = {
                "Food": [],
                "Disease": [],
                "Nutrient": [],
                "DietaryPattern": []
            }
            
            # 调用NER进行重新识别
            ner_result = run_ner_with_reflection(
                pmid=request.pmid,
                abstract=request.abstract,
                seed_entities=seed_entities,
                topic=topic,
                reflection_report=report
            )
            
            new_entities = ner_result.get('entities', [])
            
            if not new_entities:
                logger.warning("NER re-extraction returned no entities")
                return {
                    'pmid': request.pmid,
                    'triple_idx': request.triple_idx,
                    'feedback_report': report,
                    're_extracted': False,
                    'reason': 'NER returned no entities'
                }
            
            logger.info(f"NER re-extraction returned {len(new_entities)} entities")
            
            # 2. 重新执行RE
            logger.info(f"Step 2: Re-running RE with new entities...")
            
            re_result = run_re_with_reflection(
                pmid=request.pmid,
                abstract=request.abstract,
                entities=new_entities,
                topic=topic,
                reflection_report=report
            )
            
            new_triples = re_result.get('triples', [])
            
            if not new_triples:
                logger.warning("RE re-extraction returned no triples")
                return {
                    'pmid': request.pmid,
                    'triple_idx': request.triple_idx,
                    'feedback_report': report,
                    're_extracted': False,
                    'reason': 'RE returned no triples'
                }
            
            logger.info(f"RE re-extraction returned {len(new_triples)} triples")
            
            # 3. 找到与原始三元组最匹配的新三元组
            best_match = self._find_best_matching_triple(new_triples, original_triple)
            
            if best_match:
                logger.info(f"Found matching triple: {best_match.get('subject')} - {best_match.get('relation')} - {best_match.get('object')}")
                
                # 标记is_retry=True，防止死循环
                best_match['is_retry'] = True
                best_match['feedback_source'] = 'feedback_loop_2'
                best_match['original_entity'] = request.feedback_entity_text
                
                return {
                    'pmid': request.pmid,
                    'triple_idx': request.triple_idx,
                    'feedback_report': report,
                    're_extracted': True,
                    'new_triple': best_match,
                    'all_new_triples': new_triples
                }
            else:
                logger.warning("No matching triple found in re-extraction results")
                return {
                    'pmid': request.pmid,
                    'triple_idx': request.triple_idx,
                    'feedback_report': report,
                    're_extracted': True,
                    'new_triple': None,
                    'all_new_triples': new_triples,
                    'reason': 'No matching triple found'
                }
                
        except Exception as e:
            logger.error(f"Feedback Loop ② failed: {e}", exc_info=True)
            return {
                'pmid': request.pmid,
                'triple_idx': request.triple_idx,
                'feedback_report': report,
                're_extracted': False,
                'reason': f'Error: {str(e)}'
            }
    
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
        original_relation = original_triple.get('relation', '').lower()
        
        best_match = None
        best_score = 0
        
        for triple in new_triples:
            new_subject = triple.get('subject', '').lower()
            new_object = triple.get('object', '').lower()
            new_relation = triple.get('relation', '').lower()
            
            # 计算匹配分数
            score = 0
            
            # 实体匹配（更宽松的条件：包含关系）
            if original_subject in new_subject or new_subject in original_subject:
                score += 2
            if original_object in new_object or new_object in original_object:
                score += 2
            
            # 关系匹配
            if original_relation == new_relation:
                score += 1
            
            # 优先选择置信度高的
            score += triple.get('confidence', 0) * 0.5
            
            if score > best_score:
                best_score = score
                best_match = triple
        
        # 只有当匹配分数足够高时才返回
        if best_score >= 2:
            return best_match
        
        return None
    
    def process_feedback(self):
        """Process all feedback requests
        
        反馈回路②完整流程：
        1. 加载反馈请求
        2. 调用Extraction Agent重新抽取（NER+RE）
        3. 调用Grounding Agent重新对齐（is_retry=True）
        4. 合并结果
        """
        if not self.feedback_requests:
            logger.info("No feedback requests to process")
            return
        
        logger.info("=" * 70)
        logger.info(f"Starting to process {len(self.feedback_requests)} feedback requests")
        logger.info("=" * 70)
        
        processed_results = []
        
        for idx, request in enumerate(self.feedback_requests):
            logger.info(f"\nProcessing feedback request {idx + 1}/{len(self.feedback_requests)}")
            logger.info(f"  PMID: {request.pmid}")
            logger.info(f"  Entity: {request.feedback_entity_text}")
            
            # Step 1: 调用Extraction Agent进行反馈重抽取
            re_extraction_result = self.call_extraction_agent_with_feedback(request)
            
            if re_extraction_result and re_extraction_result.get('re_extracted'):
                new_triple = re_extraction_result.get('new_triple')
                
                if new_triple:
                    # Step 2: 调用Grounding Agent重新对齐（is_retry=True）
                    logger.info(f"Step 3: Re-running Grounding Agent with is_retry=True...")
                    
                    aligned_triple = self._re_align_triple(new_triple, request.abstract)
                    
                    if aligned_triple:
                        re_extraction_result['aligned_triple'] = aligned_triple
                        logger.info(f"Re-alignment completed: status={aligned_triple.get('head_alignment', {}).get('status', 'unknown')}")
                    else:
                        logger.warning("Re-alignment failed")
            
            if re_extraction_result:
                processed_results.append(re_extraction_result)
        
        # Merge results
        self.merge_results(processed_results)
        
        logger.info("=" * 70)
        logger.info("Feedback Loop ② Processing Completed")
        logger.info("=" * 70)
    
    def _re_align_triple(self, triple: Dict, abstract: str) -> Optional[Dict]:
        """重新对齐三元组（is_retry=True）
        
        Args:
            triple: 重抽取后的三元组
            abstract: 原始摘要
            
        Returns:
            Optional[Dict]: 对齐后的三元组
        """
        try:
            grounding_agent = self._get_grounding_agent()
            
            # 对齐三元组（is_retry=True）
            aligned_triple = grounding_agent.ground_single_triple(
                triple, abstract, is_retry=True
            )
            
            return aligned_triple
            
        except Exception as e:
            logger.error(f"Re-alignment failed: {e}", exc_info=True)
            return None
    
    def merge_results(self, processed_results: List[Dict]):
        """Merge feedback processing results
        
        Args:
            processed_results: Processed results
        """
        # Start with original results
        if self.original_results:
            self.updated_results = self.original_results.copy()
        else:
            self.updated_results = {
                'aligned_triples': [],
                'feedback_requests': [],
                'timestamp': time.strftime("%Y%m%d_%H%M%S")
            }
        
        # 替换处理成功的三元组
        replaced_count = 0
        for result in processed_results:
            triple_idx = result.get('triple_idx')
            aligned_triple = result.get('aligned_triple')
            
            if aligned_triple is not None and triple_idx is not None:
                # 找到并替换对应的三元组
                if 'aligned_triples' in self.updated_results:
                    triples = self.updated_results['aligned_triples']
                    if 0 <= triple_idx < len(triples):
                        # 保存原始三元组的索引，然后替换
                        triples[triple_idx] = aligned_triple
                        replaced_count = replaced_count + 1
        
        # Add feedback processing information
        self.updated_results['feedback_processed'] = processed_results
        self.updated_results['feedback_completed'] = True
        
        logger.info(f"Merged {len(processed_results)} feedback processing results, replaced {replaced_count} triples")
    
    def save_updated_results(self):
        """Save updated results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        with open(GROUNDING_ALIGNMENT_LATEST_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.updated_results, f, ensure_ascii=False, indent=2)
        
        json_archive = os.path.join(GROUNDING_ALIGNMENT_DIR, f"grounding_alignment_after_feedback_{timestamp}.json")
        with open(json_archive, 'w', encoding='utf-8') as f:
            json.dump(self.updated_results, f, ensure_ascii=False, indent=2)
        
        # Save Excel (simplified version)
        if 'aligned_triples' in self.updated_results:
            excel_rows = []
            for triple in self.updated_results['aligned_triples']:
                head_align = triple.get('head_alignment', {})
                tail_align = triple.get('tail_alignment', {})
                
                row = {
                    'pmid': triple.get('pmid', ''),
                    'subject': triple.get('subject', ''),
                    'relation': triple.get('relation', ''),
                    'object': triple.get('object', ''),
                    'source_cluster': triple.get('source_cluster', ''),
                    'quality_tier': triple.get('quality_tier', ''),
                    'Score_t': triple.get('Score_t', ''),
                    'head_status': head_align.get('status', ''),
                    'head_standard': head_align.get('standard_name', ''),
                    'head_similarity': head_align.get('similarity', 0),
                    'tail_status': tail_align.get('status', ''),
                    'tail_standard': tail_align.get('standard_name', ''),
                    'tail_similarity': tail_align.get('similarity', 0),
                    'needs_feedback': triple.get('needs_feedback', False),
                    'feedback_completed': True
                }
                excel_rows.append(row)
            
            df_excel = pd.DataFrame(excel_rows)
            df_excel.to_excel(GROUNDING_ALIGNMENT_LATEST, index=False)
            
            excel_archive = os.path.join(GROUNDING_ALIGNMENT_DIR, f"grounding_alignment_after_feedback_{timestamp}.xlsx")
            df_excel.to_excel(excel_archive, index=False)
        
        logger.info("=" * 60)
        logger.info("Updated Results Saved")
        logger.info(f"  - JSON: {GROUNDING_ALIGNMENT_LATEST_JSON}")
        logger.info(f"  - Excel: {GROUNDING_ALIGNMENT_LATEST}")
        logger.info("=" * 60)
    
    def cleanup_feedback_file(self):
        """Clean up feedback request file"""
        if os.path.exists(FEEDBACK_REFINEMENT_REQUESTS):
            os.remove(FEEDBACK_REFINEMENT_REQUESTS)
            logger.info("Deleted feedback request file")
    
    def run(self):
        """Run Feedback Loop ②"""
        logger.info("=" * 70)
        logger.info("Feedback Loop ② Starting")
        logger.info("=" * 70)
        
        # 1. Load feedback requests
        has_requests = self.load_feedback_requests()
        if not has_requests:
            return
        
        # 2. Load original results
        self.load_original_results()
        
        # 3. Process feedback
        self.process_feedback()
        
        # 4. Save updated results
        self.save_updated_results()
        
        # 5. Clean up feedback file
        self.cleanup_feedback_file()
        
        logger.info("=" * 70)
        logger.info("Feedback Loop ② Execution Completed")
        logger.info("=" * 70)


def main():
    """Main function"""
    controller = FeedbackLoopController()
    controller.run()


if __name__ == "__main__":
    main()



