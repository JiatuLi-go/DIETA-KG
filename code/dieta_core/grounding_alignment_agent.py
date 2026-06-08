#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grounding_alignment.py - Grounding Agent（语义锚定智能体）：实体标准化对齐与属性补全

功能：
- 分层异构对齐策略：不同类型实体使用不同对齐工具
- 疾病实体对齐：Disease Ontology + SapBERT
- 食物与营养素对齐：自建本体 + FAISS + all-mpnet-base-v2
- 饮食模式规范化：MeSH/ONS 词表优先 + 规则后备
- LLM语义验证与反馈回路②

输入：
- results/quality_decision/quality_decision_latest.xlsx
- results/quality_decision/quality_decision_latest.json（用于反馈回路②）

输出：
- results/grounding_alignment/grounding_alignment_latest.xlsx（固定名，每次覆盖）
- results/grounding_alignment/grounding_alignment_[时间戳].xlsx（带时间戳归档）
- results/grounding_alignment/grounding_alignment_latest.json（完整结果，含对齐信息）
"""

import os
import time
import json
import re
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import faiss
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from settings import (
    API_KEY, API_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    API_RETRY_TIMES, API_RETRY_INTERVAL,
    QUALITY_DECISION_LATEST_XLSX, QUALITY_DECISION_LATEST_JSON,
    GROUNDING_ALIGNMENT_DIR, GROUNDING_ALIGNMENT_LATEST, GROUNDING_ALIGNMENT_LATEST_JSON, FEEDBACK_REFINEMENT_REQUESTS,
    GROUNDING_DATA_DIR, NUTRIENT_TRIPLETS_FILE, HUMANIDO_FILE, INDEX_DIR,
    FOOD_MODEL
)
from model_client import LLMClient
from utils import setup_logging

# 配置日志
logger = setup_logging(GROUNDING_ALIGNMENT_DIR, "grounding_alignment")

# 确保结果目录存在
os.makedirs(GROUNDING_ALIGNMENT_DIR, exist_ok=True)


class AlignmentStatus(Enum):
    """对齐状态枚举"""
    DIRECT_HIT = "direct_hit"                      # 相似度 ≥ 0.85，直接锚定
    LLM_VERIFIED = "llm_verified"                  # [0.70, 0.85)区间，LLM验证通过
    ALIGNED_VIA_EXPANSION = "aligned_via_expansion"  # 缩写展开后匹配成功
    NEW_CONCEPT = "new_concept"                    # 合法新概念，本体未收录
    INVALID = "invalid"                            # 噪声，无效实体
    LOW_CONFIDENCE = "low_confidence"              # 低置信度（食物/营养素）
    OOV = "oov"                                    # 旧状态，保留兼容
    FEEDBACK_REQUESTED = "feedback_requested"      # 触发反馈回路②


@dataclass
class AlignmentResult:
    """对齐结果数据类"""
    original_entity: str
    entity_type: str
    status: AlignmentStatus
    similarity: float = 0.0
    standard_id: Optional[str] = None
    standard_name: Optional[str] = None
    definition: Optional[str] = None
    synonyms: List[str] = None
    xrefs: List[str] = None
    attributes: Dict[str, Any] = None
    llm_verification: Optional[bool] = None
    is_retry: bool = False
    # 开放世界验证新增字段
    is_valid: Optional[bool] = None           # LLM判断是否为合法概念
    is_abbreviation: Optional[bool] = None    # 是否为缩写
    full_form: Optional[str] = None           # 缩写的完整形式
    is_new_concept: Optional[bool] = None     # 是否为新概念
    description: Optional[str] = None         # 新概念的描述
    reason: Optional[str] = None              # LLM判断依据
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if self.xrefs is None:
            self.xrefs = []
        if self.attributes is None:
            self.attributes = {}


class DiseaseOntologyLoader:
    """加载和解析Disease Ontology (HumanDO.json)"""
    
    def __init__(self, do_file: str):
        self.do_file = do_file
        self.disease_nodes = []
        self.disease_texts = []
        self.disease_ids = []
        self.disease_labels = []
        self.disease_synonyms = []
        self.disease_definitions = []
        self.disease_xrefs = []
        
    def load(self):
        """加载HumanDO.json"""
        logger.info(f"正在加载 Disease Ontology: {self.do_file}")
        with open(self.do_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'graphs' in data and len(data['graphs']) > 0:
            graph = data['graphs'][0]
            if 'nodes' in graph:
                self.disease_nodes = [
                    n for n in graph['nodes'] 
                    if n.get('type') == 'CLASS'
                ]
        
        logger.info(f"已加载 {len(self.disease_nodes)} 个疾病节点")
        
        # 预处理文本
        for node in self.disease_nodes:
            node_id = node.get('id', '')
            label = node.get('lbl', '')
            
            # 提取同义词
            synonyms = []
            meta = node.get('meta', {})
            if 'synonyms' in meta:
                for syn in meta['synonyms']:
                    if 'val' in syn:
                        synonyms.append(syn['val'])
            
            # 提取定义
            definition = ''
            if 'definition' in meta and 'val' in meta['definition']:
                definition = meta['definition']['val']
            
            # 提取xrefs
            xrefs = []
            if 'xrefs' in meta:
                for xref in meta['xrefs']:
                    if 'val' in xref:
                        xrefs.append(xref['val'])
            
            # 构建用于编码的文本：label + 所有同义词
            text_parts = [label] + synonyms
            combined_text = ' '.join(text_parts)
            
            self.disease_texts.append(combined_text)
            self.disease_ids.append(node_id)
            self.disease_labels.append(label)
            self.disease_synonyms.append(synonyms)
            self.disease_definitions.append(definition)
            self.disease_xrefs.append(xrefs)
        
        logger.info("Disease Ontology 预处理完成")


class DiseaseAligner:
    """疾病实体对齐器：Disease Ontology + SapBERT"""
    
    DISEASE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grounding_data", "index", "disease")
    DISEASE_EMBEDDINGS_FILE = os.path.join(DISEASE_CACHE_DIR, "disease_embeddings.npy")
    DISEASE_META_FILE = os.path.join(DISEASE_CACHE_DIR, "disease_meta.pkl")
    
    def __init__(self, do_loader: DiseaseOntologyLoader = None, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        self.do_loader = do_loader
        self.model_name = model_name
        self.model = None
        self.disease_embeddings = None
        self._use_cache = False
        
        self.alpha_direct = 0.85
        self.alpha_llm = 0.70
        self.alpha_feedback = 0.55
        
    def load_from_cache(self):
        """从缓存加载预编码的疾病嵌入"""
        if os.path.exists(self.DISEASE_EMBEDDINGS_FILE) and os.path.exists(self.DISEASE_META_FILE):
            logger.info("从缓存加载预编码的疾病嵌入...")
            import pickle
            self.disease_embeddings = np.load(self.DISEASE_EMBEDDINGS_FILE)
            with open(self.DISEASE_META_FILE, 'rb') as f:
                meta = pickle.load(f)
            if self.do_loader is None:
                self.do_loader = DiseaseOntologyLoader("")
            self.do_loader.disease_ids = meta['disease_ids']
            self.do_loader.disease_labels = meta['disease_labels']
            self.do_loader.disease_definitions = meta['disease_definitions']
            self.do_loader.disease_synonyms = meta['disease_synonyms']
            self.do_loader.disease_xrefs = meta['disease_xrefs']
            self.do_loader.disease_texts = meta['disease_texts']
            self._use_cache = True
            logger.info(f"从缓存加载完成，共 {len(self.do_loader.disease_ids)} 个疾病节点")
            return True
        return False
        
    def load_model(self):
        """加载SapBERT模型"""
        logger.info(f"正在加载 SapBERT 模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, local_files_only=True)
        logger.info("SapBERT 模型加载完成")
        
    def encode_diseases(self):
        """编码所有疾病节点（优先使用缓存）"""
        if self._use_cache:
            logger.info("使用预编码的疾病嵌入，跳过编码")
            return
            
        if self.load_from_cache():
            return
            
        if self.do_loader is None:
            raise ValueError("需要 DiseaseOntologyLoader 来编码疾病节点")
            
        logger.info("正在编码 Disease Ontology 节点...")
        self.disease_embeddings = self.model.encode(
            self.do_loader.disease_texts,
            batch_size=64,
            show_progress_bar=True
        )
        norms = np.linalg.norm(self.disease_embeddings, axis=1, keepdims=True)
        self.disease_embeddings = self.disease_embeddings / norms
        logger.info(f"疾病节点编码完成，维度: {self.disease_embeddings.shape}")
        
    def align(self, entity: str, abstract: str = "", is_retry: bool = False) -> AlignmentResult:
        """对齐单个疾病实体
        
        Args:
            entity: 待对齐的疾病实体
            abstract: 原始摘要（用于LLM验证）
            is_retry: 是否为重试（防止死循环）
            
        Returns:
            AlignmentResult: 对齐结果
        """
        if self.model is None or self.disease_embeddings is None:
            self.load_model()
            self.encode_diseases()
        
        # 编码待对齐实体
        entity_embedding = self.model.encode([entity])[0]
        entity_embedding = entity_embedding / np.linalg.norm(entity_embedding)
        
        # 计算余弦相似度
        similarities = np.dot(self.disease_embeddings, entity_embedding)
        
        # 获取top-3
        top_indices = np.argsort(similarities)[::-1][:3]
        top_similarities = similarities[top_indices]
        
        best_idx = top_indices[0]
        best_similarity = top_similarities[0]
        best_label = self.do_loader.disease_labels[best_idx]
        
        result = AlignmentResult(
            original_entity=entity,
            entity_type="Disease",
            status=AlignmentStatus.OOV,
            similarity=best_similarity,
            is_retry=is_retry
        )
        
        logger.info(f"[对齐流程] 实体='{entity}', 相似度={best_similarity:.4f}, 候选='{best_label}'")
        
        if best_similarity >= self.alpha_direct:
            logger.info(f"[对齐流程] 相似度 >= 0.85 → DIRECT_HIT (直接命中)")
            result.status = AlignmentStatus.DIRECT_HIT
            result.standard_id = self.do_loader.disease_ids[best_idx]
            result.standard_name = best_label
            result.definition = self.do_loader.disease_definitions[best_idx]
            result.synonyms = self.do_loader.disease_synonyms[best_idx]
            result.xrefs = self.do_loader.disease_xrefs[best_idx]
            
        elif best_similarity >= self.alpha_llm:
            logger.info(f"[对齐流程] 0.70 <= 相似度 < 0.85 → 进入LLM语义验证阶段")
            result.standard_id = self.do_loader.disease_ids[best_idx]
            result.standard_name = best_label
            result.definition = self.do_loader.disease_definitions[best_idx]
            result.synonyms = self.do_loader.disease_synonyms[best_idx]
            result.xrefs = self.do_loader.disease_xrefs[best_idx]
            
            top_candidates = []
            for i, idx in enumerate(top_indices):
                top_candidates.append({
                    'rank': i + 1,
                    'id': self.do_loader.disease_ids[idx],
                    'name': self.do_loader.disease_labels[idx],
                    'definition': self.do_loader.disease_definitions[idx],
                    'similarity': float(similarities[idx])
                })
            
            llm_verified = self._llm_verify(entity, abstract, top_candidates)
            result.llm_verification = llm_verified
            
            if llm_verified:
                logger.info(f"[对齐流程] LLM语义验证通过 → LLM_VERIFIED")
                result.status = AlignmentStatus.LLM_VERIFIED
            else:
                logger.info(f"[对齐流程] LLM语义验证失败 → 进入开放世界验证")
                open_world_result = self._open_world_verify(entity, abstract)
                result = self._process_open_world_result(result, open_world_result, best_idx, best_label, best_similarity, entity, abstract, is_retry)
                    
        else:
            logger.info(f"[对齐流程] 相似度 < 0.70 → 进入开放世界验证阶段")
            open_world_result = self._open_world_verify(entity, abstract)
            result = self._process_open_world_result(result, open_world_result, best_idx, best_label, best_similarity, entity, abstract, is_retry)
            
        return result
    
    def _process_open_world_result(self, result: AlignmentResult, open_world_result: Dict, 
                                    best_idx: int, best_label: str, best_similarity: float,
                                    entity: str, abstract: str, is_retry: bool) -> AlignmentResult:
        """处理开放世界验证结果（符合论文3.6.2节设计）
        
        三类处置路径：
        - 路径A：合法已知概念，缩写展开后匹配 → ALIGNED_VIA_EXPANSION
        - 路径B：合法新概念，本体未收录 → NEW_CONCEPT
        - 路径C：噪声，无效实体 → INVALID
        
        反馈回路②触发条件（论文3.6.3节）：
        - is_valid=true 且 is_abbreviation=false
        - 且实体文本是候选标准名子串
        - 且相似度 >= ALPHA_BOUNDARY (0.55)
        """
        ALPHA_BOUNDARY = 0.55
        
        result.is_valid = open_world_result['is_valid']
        result.is_abbreviation = open_world_result['is_abbreviation']
        result.full_form = open_world_result.get('full_form', '')
        result.is_new_concept = open_world_result.get('is_new_concept', False)
        result.description = open_world_result.get('description', '')
        result.reason = open_world_result.get('reason', '')
        
        logger.info(f"[开放世界验证] is_valid={open_world_result['is_valid']}, "
                   f"is_abbreviation={open_world_result['is_abbreviation']}, "
                   f"is_new_concept={open_world_result.get('is_new_concept', False)}")
        
        if open_world_result['is_valid']:
            if open_world_result['is_abbreviation']:
                logger.info(f"[开放世界验证] 路径A: 缩写展开后重检索, full_form='{open_world_result.get('full_form', '')}'")
                expanded_form = open_world_result.get('full_form', '')
                if expanded_form.lower().strip() == entity.lower().strip():
                    logger.warning(f"[开放世界验证] 缩写展开失败，'{entity}' 无法展开为不同形式")
                    result.status = AlignmentStatus.NEW_CONCEPT
                    result.is_abbreviation = True
                    result.full_form = expanded_form
                    self._log_ontology_extension(entity, "Disease", open_world_result)
                else:
                    expansion_result = self._abbreviation_expansion_retrieval(expanded_form, abstract, is_retry=True)
                    if expansion_result.status in [AlignmentStatus.DIRECT_HIT, AlignmentStatus.LLM_VERIFIED]:
                        logger.info(f"[开放世界验证] 路径A成功: 缩写展开后匹配 → ALIGNED_VIA_EXPANSION")
                        expansion_result.status = AlignmentStatus.ALIGNED_VIA_EXPANSION
                        expansion_result.original_entity = entity
                        expansion_result.is_abbreviation = True
                        expansion_result.full_form = expanded_form
                        return expansion_result
                    else:
                        logger.info(f"[开放世界验证] 路径A失败: 缩写展开后仍未匹配 → NEW_CONCEPT")
                        result.status = AlignmentStatus.NEW_CONCEPT
                        result.is_valid = True
                        result.is_abbreviation = True
                        result.full_form = expanded_form
                        result.is_new_concept = True
                        result.description = open_world_result.get('description', '')
                        self._log_ontology_extension(entity, "Disease", open_world_result)
                        
            else:
                is_boundary_truncation = (best_similarity >= ALPHA_BOUNDARY and 
                                         entity.lower() in best_label.lower() and 
                                         not is_retry)
                
                if is_boundary_truncation:
                    logger.info(f"[开放世界验证] 满足反馈回路②触发条件:")
                    logger.info(f"  - is_valid=true, is_abbreviation=false")
                    logger.info(f"  - 相似度={best_similarity:.4f} >= {ALPHA_BOUNDARY}")
                    logger.info(f"  - '{entity.lower()}' in '{best_label.lower()}'")
                    logger.info(f"  → FEEDBACK_REQUESTED (触发反馈回路②)")
                    result.status = AlignmentStatus.FEEDBACK_REQUESTED
                    result.standard_id = self.do_loader.disease_ids[best_idx]
                    result.standard_name = best_label
                    result.similarity = best_similarity
                elif open_world_result['is_new_concept']:
                    logger.info(f"[开放世界验证] 路径B: 合法新概念，本体未收录 → NEW_CONCEPT")
                    result.status = AlignmentStatus.NEW_CONCEPT
                    result.is_valid = True
                    result.is_new_concept = True
                    result.description = open_world_result.get('description', '')
                    self._log_ontology_extension(entity, "Disease", open_world_result)
                else:
                    logger.info(f"[开放世界验证] 路径B: 合法概念但本体未收录 → NEW_CONCEPT")
                    result.status = AlignmentStatus.NEW_CONCEPT
                    result.is_valid = True
                    result.is_new_concept = True
                    result.description = open_world_result.get('description', '')
                    self._log_ontology_extension(entity, "Disease", open_world_result)
        else:
            logger.info(f"[开放世界验证] 路径C: 噪声/无效实体 → INVALID")
            result.status = AlignmentStatus.INVALID
            result.is_valid = False
            result.reason = open_world_result.get('reason', '')
            
        return result
    
    def _llm_verify(self, entity: str, abstract: str, candidates: List[Dict]) -> bool:
        """使用LLM进行语义验证（[0.70, 0.85)区间）
        
        Returns:
            bool: 是否验证通过
        """
        try:
            llm_client = LLMClient()
            
            candidates_str = "\n".join([
                f"Candidate {c['rank']}:\n"
                f"  Name: {c['name']}\n"
                f"  Definition: {c['definition'][:200] if c['definition'] else 'N/A'}..."
                for c in candidates
            ])
            
            prompt = f"""You are a professional medical entity alignment verification expert. Please determine if the extracted disease entity is semantically consistent with the candidate standard entity.

[Abstract Context]
{abstract[:1000]}

[Extracted Disease Entity]
{entity}

[Candidate Standard Entities (Top-3)]
{candidates_str}

Please answer the following questions (JSON output only):
1. Is the extracted entity "{entity}" semantically consistent with Candidate 1 "{candidates[0]['name']}"?
2. If yes, explain why; if no, explain why not.

Output format:
{{
    "is_match": true/false,
    "reason": "Detailed explanation"
}}
"""
            
            response = llm_client.generate(
                prompt=prompt,
                model=LLM_MODEL,
                temperature=0.1,
                max_tokens=500
            )
            
            # 解析JSON响应
            json_str = self._extract_json(response)
            if json_str:
                result = json.loads(json_str)
                return result.get('is_match', False)
            
            return False
            
        except Exception as e:
            logger.warning(f"LLM验证失败: {e}")
            return False
    
    def _open_world_verify(self, entity: str, abstract: str) -> Dict:
        """开放世界LLM实体合法性验证
        
        判断实体是否为合法概念，识别缩写、新概念、噪声
        
        Args:
            entity: 待验证实体
            abstract: 原始摘要上下文
            
        Returns:
            Dict: {
                "is_valid": true/false,
                "full_form": "完整形式（如果是缩写）",
                "is_abbreviation": true/false,
                "is_new_concept": true/false,
                "description": "简要描述（新概念时填写）",
                "reason": "判断依据"
            }
        """
        try:
            llm_client = LLMClient()
            
            prompt = f"""You are a professional biomedical domain expert. Please verify if the following entity is a legitimate concept in the disease/medical domain.

[Abstract Context]
{abstract[:1000]}

[Entity to Verify]
{entity}

Please answer the following questions (JSON output only):
1. Is this entity a legitimate concept in the disease/medical domain?
2. If it is an abbreviation, please provide the full form.
3. If it is a new concept not yet in standard ontologies, please provide a brief description.

Output format:
{{
    "is_valid": true/false,
    "is_abbreviation": true/false,
    "full_form": "Full form if abbreviation, otherwise empty string",
    "is_new_concept": true/false,
    "description": "Brief description if new concept, otherwise empty string",
    "reason": "Detailed reasoning for the judgment"
}}
"""
            
            response = llm_client.generate(
                prompt=prompt,
                model=LLM_MODEL,
                temperature=0.1,
                max_tokens=500
            )
            
            # 解析JSON响应
            json_str = self._extract_json(response)
            if json_str:
                result = json.loads(json_str)
                return result
            
            return {
                "is_valid": False,
                "is_abbreviation": False,
                "full_form": "",
                "is_new_concept": False,
                "description": "",
                "reason": "Failed to parse LLM response"
            }
            
        except Exception as e:
            logger.warning(f"开放世界验证失败: {e}")
            return {
                "is_valid": False,
                "is_abbreviation": False,
                "full_form": "",
                "is_new_concept": False,
                "description": "",
                "reason": f"Error: {str(e)}"
            }
    
    def _abbreviation_expansion_retrieval(self, full_form: str, abstract: str, is_retry: bool) -> AlignmentResult:
        """缩写展开后重新检索
        
        Args:
            full_form: 缩写的完整形式
            abstract: 原始摘要
            is_retry: 是否为重试
            
        Returns:
            AlignmentResult: 重新检索的结果
        """
        logger.info(f"缩写展开重检索: {full_form}")
        return self.align(full_form, abstract, is_retry)
    
    def _log_ontology_extension(self, entity: str, entity_type: str, verification_result: Dict):
        """写入本体扩展日志
        
        Args:
            entity: 实体名称
            entity_type: 实体类型
            verification_result: LLM验证结果
        """
        import csv
        from datetime import datetime
        
        log_file = os.path.join(GROUNDING_ALIGNMENT_DIR, "ontology_extension_candidates.csv")
        
        # 检查文件是否存在，决定是否写入header
        file_exists = os.path.exists(log_file)
        
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'entity', 'entity_type', 'is_new_concept', 
                    'description', 'reason', 'status'
                ])
            writer.writerow([
                datetime.now().isoformat(),
                entity,
                entity_type,
                verification_result.get('is_new_concept', False),
                verification_result.get('description', ''),
                verification_result.get('reason', ''),
                'pending_review'
            ])
        
        logger.info(f"已记录本体扩展候选: {entity} ({entity_type})")
    
    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取JSON"""
        text = text.strip()
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace+1]
        return None


class FoodNutrientAligner:
    """食物与营养素对齐器：自建本体 + FAISS
    
    根据论文设计：
    - 相似度 ≥ 0.85：直接锚定（DIRECT_HIT）
    - 相似度 [0.70, 0.85)：LLM语义验证
    - 相似度 < 0.70：开放世界验证
    """
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        
        # 加载食物索引
        self.food_index = None
        self.food_texts = None
        self.food_attrs = None
        
        # 加载营养素索引
        self.nutrient_index = None
        self.nutrient_texts = None
        self.nutrient_attrs = None
        
        # 加载模型
        self.model = None
        
        # 阈值配置（符合论文设计）
        self.alpha_direct = 0.85      # 直接锚定阈值
        self.alpha_llm = 0.70         # LLM验证阈值
        self.alpha_boundary = 0.55    # 反馈回路②边界截断检测阈值
        
    def load(self):
        """加载索引和模型"""
        logger.info("正在加载食物和营养素索引...")
        
        # 加载食物索引
        food_index_path = os.path.join(self.index_dir, "food_desc.faiss")
        food_texts_path = os.path.join(self.index_dir, "food_desc.pkl")
        food_attrs_path = os.path.join(self.index_dir, "food_attr.pkl")
        
        if os.path.exists(food_index_path):
            self.food_index = faiss.read_index(food_index_path)
            with open(food_texts_path, 'rb') as f:
                self.food_texts = pickle.load(f)
            with open(food_attrs_path, 'rb') as f:
                self.food_attrs = pickle.load(f)
            logger.info(f"食物索引加载完成: {len(self.food_texts)} 个条目")
        
        # 加载营养素索引
        nutrient_index_path = os.path.join(self.index_dir, "nutrient_obj.faiss")
        nutrient_texts_path = os.path.join(self.index_dir, "nutrient_obj.pkl")
        nutrient_attrs_path = os.path.join(self.index_dir, "nutrient_attr.pkl")
        
        if os.path.exists(nutrient_index_path):
            self.nutrient_index = faiss.read_index(nutrient_index_path)
            with open(nutrient_texts_path, 'rb') as f:
                self.nutrient_texts = pickle.load(f)
            with open(nutrient_attrs_path, 'rb') as f:
                self.nutrient_attrs = pickle.load(f)
            logger.info(f"营养素索引加载完成: {len(self.nutrient_texts)} 个条目")
        
        logger.info(f"正在加载模型: {FOOD_MODEL}")
        self.model = SentenceTransformer(FOOD_MODEL, local_files_only=True)
        logger.info("模型加载完成")
    
    def align_food(self, entity: str, abstract: str = "", is_retry: bool = False) -> AlignmentResult:
        """对齐食物实体
        
        Args:
            entity: 待对齐的食物实体
            abstract: 原始摘要（用于LLM验证和开放世界验证）
            is_retry: 是否为重试（防止死循环）
            
        Returns:
            AlignmentResult: 对齐结果
        """
        if self.food_index is None:
            self.load()
        
        result = AlignmentResult(
            original_entity=entity,
            entity_type="Food",
            status=AlignmentStatus.OOV,
            is_retry=is_retry
        )
        
        embedding = self.model.encode([entity])[0]
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        D, I = self.food_index.search(embedding.reshape(1, -1), 3)
        similarities = D[0]
        indices = I[0]
        
        best_similarity = similarities[0]
        best_idx = indices[0]
        best_name = self.food_texts[best_idx]
        
        result.similarity = float(best_similarity)
        
        logger.info(f"[对齐流程-Food] 实体='{entity}', 相似度={best_similarity:.4f}, 候选='{best_name}'")
        
        if best_similarity >= self.alpha_direct:
            logger.info(f"[对齐流程-Food] 相似度 >= 0.85 → DIRECT_HIT (直接命中)")
            result.status = AlignmentStatus.DIRECT_HIT
            result.standard_name = best_name
            result.attributes = self.food_attrs[best_idx]
            
        elif best_similarity >= self.alpha_llm:
            logger.info(f"[对齐流程-Food] 0.70 <= 相似度 < 0.85 → 进入LLM语义验证阶段")
            result.standard_name = best_name
            result.attributes = self.food_attrs[best_idx]
            
            top_candidates = []
            for i, (sim, idx) in enumerate(zip(similarities, indices)):
                top_candidates.append({
                    'rank': i + 1,
                    'name': self.food_texts[idx],
                    'similarity': float(sim)
                })
            
            llm_verified = self._llm_verify_food_nutrient(entity, abstract, top_candidates, "Food")
            result.llm_verification = llm_verified
            
            if llm_verified:
                logger.info(f"[对齐流程-Food] LLM语义验证通过 → LLM_VERIFIED")
                result.status = AlignmentStatus.LLM_VERIFIED
            else:
                logger.info(f"[对齐流程-Food] LLM语义验证失败 → 进入开放世界验证")
                open_world_result = self._open_world_verify_food_nutrient(entity, abstract, "Food")
                result = self._process_open_world_result_food_nutrient(
                    result, open_world_result, best_idx, best_name, best_similarity, entity, abstract, is_retry, "Food"
                )
                    
        else:
            logger.info(f"[对齐流程-Food] 相似度 < 0.70 → 进入开放世界验证阶段")
            open_world_result = self._open_world_verify_food_nutrient(entity, abstract, "Food")
            result = self._process_open_world_result_food_nutrient(
                result, open_world_result, best_idx, best_name, best_similarity, entity, abstract, is_retry, "Food"
            )
            
        return result
    
    def _process_open_world_result_food_nutrient(self, result: AlignmentResult, open_world_result: Dict,
                                                  best_idx: int, best_name: str, best_similarity: float,
                                                  entity: str, abstract: str, is_retry: bool, 
                                                  entity_type: str) -> AlignmentResult:
        """处理开放世界验证结果（食物/营养素，符合论文3.6.2节设计）
        
        三类处置路径：
        - 路径A：合法已知概念，缩写展开后匹配 → ALIGNED_VIA_EXPANSION
        - 路径B：合法新概念，本体未收录 → NEW_CONCEPT
        - 路径C：噪声，无效实体 → INVALID
        
        反馈回路②触发条件（论文3.6.3节）：
        - is_valid=true 且 is_abbreviation=false
        - 且实体文本是候选标准名子串
        - 且相似度 >= ALPHA_BOUNDARY (0.55)
        """
        ALPHA_BOUNDARY = 0.55
        
        result.is_valid = open_world_result['is_valid']
        result.is_abbreviation = open_world_result['is_abbreviation']
        result.full_form = open_world_result.get('full_form', '')
        result.is_new_concept = open_world_result.get('is_new_concept', False)
        result.description = open_world_result.get('description', '')
        result.reason = open_world_result.get('reason', '')
        
        logger.info(f"[开放世界验证-{entity_type}] is_valid={open_world_result['is_valid']}, "
                   f"is_abbreviation={open_world_result['is_abbreviation']}, "
                   f"is_new_concept={open_world_result.get('is_new_concept', False)}")
        
        if open_world_result['is_valid']:
            if open_world_result['is_abbreviation']:
                logger.info(f"[开放世界验证-{entity_type}] 路径A: 缩写展开后重检索, full_form='{open_world_result.get('full_form', '')}'")
                expanded_form = open_world_result.get('full_form', '')
                if expanded_form.lower().strip() == entity.lower().strip():
                    logger.warning(f"[开放世界验证-{entity_type}] 缩写展开失败，'{entity}' 无法展开为不同形式")
                    result.status = AlignmentStatus.NEW_CONCEPT
                    result.is_abbreviation = True
                    result.full_form = expanded_form
                    self._log_ontology_extension(entity, entity_type, open_world_result)
                else:
                    if entity_type == "Food":
                        expansion_result = self.align_food(expanded_form, abstract, is_retry=True)
                    else:
                        expansion_result = self.align_nutrient(expanded_form, abstract, is_retry=True)
                    if expansion_result.status in [AlignmentStatus.DIRECT_HIT, AlignmentStatus.LLM_VERIFIED]:
                        logger.info(f"[开放世界验证-{entity_type}] 路径A成功: 缩写展开后匹配 → ALIGNED_VIA_EXPANSION")
                        expansion_result.status = AlignmentStatus.ALIGNED_VIA_EXPANSION
                        expansion_result.original_entity = entity
                        expansion_result.is_abbreviation = True
                        expansion_result.full_form = expanded_form
                        return expansion_result
                    else:
                        logger.info(f"[开放世界验证-{entity_type}] 路径A失败: 缩写展开后仍未匹配 → NEW_CONCEPT")
                        result.status = AlignmentStatus.NEW_CONCEPT
                        result.is_valid = True
                        result.is_abbreviation = True
                        result.full_form = expanded_form
                        result.is_new_concept = True
                        result.description = open_world_result.get('description', '')
                        self._log_ontology_extension(entity, entity_type, open_world_result)
                        
            else:
                is_boundary_truncation = (best_similarity >= ALPHA_BOUNDARY and 
                                         entity.lower() in best_name.lower() and 
                                         not is_retry)
                
                if is_boundary_truncation:
                    logger.info(f"[开放世界验证-{entity_type}] 满足反馈回路②触发条件:")
                    logger.info(f"  - is_valid=true, is_abbreviation=false")
                    logger.info(f"  - 相似度={best_similarity:.4f} >= {ALPHA_BOUNDARY}")
                    logger.info(f"  - '{entity.lower()}' in '{best_name.lower()}'")
                    logger.info(f"  → FEEDBACK_REQUESTED (触发反馈回路②)")
                    result.status = AlignmentStatus.FEEDBACK_REQUESTED
                    result.standard_name = best_name
                    result.similarity = best_similarity
                elif open_world_result['is_new_concept']:
                    logger.info(f"[开放世界验证-{entity_type}] 路径B: 合法新概念，本体未收录 → NEW_CONCEPT")
                    result.status = AlignmentStatus.NEW_CONCEPT
                    result.is_valid = True
                    result.is_new_concept = True
                    result.description = open_world_result.get('description', '')
                    self._log_ontology_extension(entity, entity_type, open_world_result)
                else:
                    logger.info(f"[开放世界验证-{entity_type}] 路径B: 合法概念但本体未收录 → NEW_CONCEPT")
                    result.status = AlignmentStatus.NEW_CONCEPT
                    result.is_valid = True
                    result.is_new_concept = True
                    result.description = open_world_result.get('description', '')
                    self._log_ontology_extension(entity, entity_type, open_world_result)
        else:
            logger.info(f"[开放世界验证-{entity_type}] 路径C: 噪声/无效实体 → INVALID")
            result.status = AlignmentStatus.INVALID
            result.is_valid = False
            result.reason = open_world_result.get('reason', '')
            
        return result
    
    def align_nutrient(self, entity: str, abstract: str = "", is_retry: bool = False) -> AlignmentResult:
        """对齐营养素实体
        
        Args:
            entity: 待对齐的营养素实体
            abstract: 原始摘要（用于LLM验证和开放世界验证）
            is_retry: 是否为重试（防止死循环）
            
        Returns:
            AlignmentResult: 对齐结果
        """
        if self.nutrient_index is None:
            self.load()
        
        result = AlignmentResult(
            original_entity=entity,
            entity_type="Nutrient",
            status=AlignmentStatus.OOV,
            is_retry=is_retry
        )
        
        embedding = self.model.encode([entity])[0]
        embedding = embedding.astype('float32')
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        D, I = self.nutrient_index.search(embedding.reshape(1, -1), 3)
        similarities = D[0]
        indices = I[0]
        
        best_similarity = similarities[0]
        best_idx = indices[0]
        best_name = self.nutrient_texts[best_idx]
        
        result.similarity = float(best_similarity)
        
        logger.info(f"[对齐流程-Nutrient] 实体='{entity}', 相似度={best_similarity:.4f}, 候选='{best_name}'")
        
        if best_similarity >= self.alpha_direct:
            logger.info(f"[对齐流程-Nutrient] 相似度 >= 0.85 → DIRECT_HIT (直接命中)")
            result.status = AlignmentStatus.DIRECT_HIT
            result.standard_name = best_name
            result.attributes = self.nutrient_attrs[best_idx]
            
        elif best_similarity >= self.alpha_llm:
            logger.info(f"[对齐流程-Nutrient] 0.70 <= 相似度 < 0.85 → 进入LLM语义验证阶段")
            result.standard_name = best_name
            result.attributes = self.nutrient_attrs[best_idx]
            
            top_candidates = []
            for i, (sim, idx) in enumerate(zip(similarities, indices)):
                top_candidates.append({
                    'rank': i + 1,
                    'name': self.nutrient_texts[idx],
                    'similarity': float(sim)
                })
            
            llm_verified = self._llm_verify_food_nutrient(entity, abstract, top_candidates, "Nutrient")
            result.llm_verification = llm_verified
            
            if llm_verified:
                logger.info(f"[对齐流程-Nutrient] LLM语义验证通过 → LLM_VERIFIED")
                result.status = AlignmentStatus.LLM_VERIFIED
            else:
                logger.info(f"[对齐流程-Nutrient] LLM语义验证失败 → 进入开放世界验证")
                open_world_result = self._open_world_verify_food_nutrient(entity, abstract, "Nutrient")
                result = self._process_open_world_result_food_nutrient(
                    result, open_world_result, best_idx, best_name, best_similarity, entity, abstract, is_retry, "Nutrient"
                )
                    
        else:
            logger.info(f"[对齐流程-Nutrient] 相似度 < 0.70 → 进入开放世界验证阶段")
            open_world_result = self._open_world_verify_food_nutrient(entity, abstract, "Nutrient")
            result = self._process_open_world_result_food_nutrient(
                result, open_world_result, best_idx, best_name, best_similarity, entity, abstract, is_retry, "Nutrient"
            )
            
        return result
    
    def _llm_verify_food_nutrient(self, entity: str, abstract: str, candidates: List[Dict], entity_type: str) -> bool:
        """LLM语义验证（食物/营养素）- [0.70, 0.85)区间
        
        Args:
            entity: 待验证实体
            abstract: 原始摘要
            candidates: top-3候选列表
            entity_type: 实体类型
            
        Returns:
            bool: 是否验证通过
        """
        try:
            llm_client = LLMClient()
            
            candidates_str = "\n".join([
                f"Candidate {c['rank']}:\n"
                f"  Name: {c['name']}\n"
                f"  Similarity: {c['similarity']:.3f}"
                for c in candidates
            ])
            
            prompt = f"""You are a professional nutrition science expert. Please determine if the extracted {entity_type.lower()} entity is semantically consistent with the candidate standard entity.

[Abstract Context]
{abstract[:1000]}

[Extracted {entity_type} Entity]
{entity}

[Candidate Standard Entities (Top-3)]
{candidates_str}

Please answer the following questions (JSON output only):
1. Is the extracted entity "{entity}" semantically consistent with Candidate 1 "{candidates[0]['name']}"?
2. If yes, explain why; if no, explain why not.

Output format:
{{
    "is_match": true/false,
    "reason": "Detailed explanation"
}}
"""
            
            response = llm_client.generate(
                prompt=prompt,
                model=LLM_MODEL,
                temperature=0.1,
                max_tokens=500
            )
            
            json_str = self._extract_json(response)
            if json_str:
                result = json.loads(json_str)
                return result.get('is_match', False)
            
            return False
            
        except Exception as e:
            logger.warning(f"LLM验证失败: {e}")
            return False
    
    def _open_world_verify_food_nutrient(self, entity: str, abstract: str, entity_type: str) -> Dict:
        """开放世界LLM实体合法性验证（食物/营养素）
        
        Args:
            entity: 待验证实体
            abstract: 原始摘要上下文
            entity_type: 实体类型（Food/Nutrient）
            
        Returns:
            Dict: 验证结果
        """
        try:
            llm_client = LLMClient()
            
            prompt = f"""You are a professional nutrition science expert. Please verify if the following entity is a legitimate concept in the {entity_type.lower()} domain.

[Abstract Context]
{abstract[:1000]}

[Entity to Verify]
{entity}

[Entity Type]
{entity_type}

Please answer the following questions (JSON output only):
1. Is this entity a legitimate concept in the {entity_type.lower()} domain?
2. If it is an abbreviation, please provide the full form.
3. If it is a new concept not yet in standard databases, please provide a brief description.

Output format:
{{
    "is_valid": true/false,
    "is_abbreviation": true/false,
    "full_form": "Full form if abbreviation, otherwise empty string",
    "is_new_concept": true/false,
    "description": "Brief description if new concept, otherwise empty string",
    "reason": "Detailed reasoning for the judgment"
}}
"""
            
            response = llm_client.generate(
                prompt=prompt,
                model=LLM_MODEL,
                temperature=0.1,
                max_tokens=500
            )
            
            json_str = self._extract_json(response)
            if json_str:
                result = json.loads(json_str)
                return result
            
            return {
                "is_valid": False,
                "is_abbreviation": False,
                "full_form": "",
                "is_new_concept": False,
                "description": "",
                "reason": "Failed to parse LLM response"
            }
            
        except Exception as e:
            logger.warning(f"开放世界验证失败: {e}")
            return {
                "is_valid": False,
                "is_abbreviation": False,
                "full_form": "",
                "is_new_concept": False,
                "description": "",
                "reason": f"Error: {str(e)}"
            }
    
    def _log_ontology_extension(self, entity: str, entity_type: str, verification_result: Dict):
        """写入本体扩展日志"""
        import csv
        from datetime import datetime
        
        log_file = os.path.join(GROUNDING_ALIGNMENT_DIR, "ontology_extension_candidates.csv")
        file_exists = os.path.exists(log_file)
        
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'entity', 'entity_type', 'is_new_concept', 
                    'description', 'reason', 'status'
                ])
            writer.writerow([
                datetime.now().isoformat(),
                entity,
                entity_type,
                verification_result.get('is_new_concept', False),
                verification_result.get('description', ''),
                verification_result.get('reason', ''),
                'pending_review'
            ])
        
        logger.info(f"已记录本体扩展候选: {entity} ({entity_type})")
    
    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取JSON"""
        text = text.strip()
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace+1]
        return None


class DietaryPatternNormalizer:
    """饮食模式规范化器
    
    优先使用基于官方资源整理的 DietaryPattern 词表进行 exact/synonym 对齐；
    若无可靠命中，则回退到规则规范化并作为新节点纳入 GCDI-KG。
    """
    
    def __init__(self):
        # 预构建的缩写词表
        self.abbreviation_map = {
            "DASH": "Dietary Approaches to Stop Hypertension",
            "Med": "Mediterranean",
            "Keto": "Ketogenic",
            "Veg": "Vegetarian",
            "Vegan": "Vegan",
            "LC": "Low Carbohydrate",
            "LF": "Low Fat",
            "HP": "High Protein"
        }
        
        # 冠词列表
        self.articles = {"a", "an", "the"}
        
        # 经核实的饮食模式对齐资源
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.lexicon_file = os.path.join(
            current_dir,
            "grounding_data",
            "dietary_pattern",
            "dietary_pattern_reference_terms.json"
        )
        self.lexicon = []
        self.alias_index = {}
        self._load_lexicon()
    
    def _normalize_lookup_text(self, text: str) -> str:
        """用于词表查找的保守规范化"""
        text = text.lower()
        text = text.replace("&", " and ")
        text = re.sub(r"[‐‑‒–—−]", "-", text)
        text = text.replace("/", " ")
        text = text.replace(",", " ")
        text = text.replace("-", " ")
        text = re.sub(r"[^a-z0-9()\s]", " ", text)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _remove_parenthetical(self, text: str) -> str:
        """移除括号内容，用于保守的别名回收"""
        text = re.sub(r"\([^)]*\)", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    
    def _singularize_terminal_diets(self, text: str) -> str:
        """将句尾复数 diets 保守地回收到 diet"""
        return re.sub(r"\bdiets\b", "diet", text, flags=re.IGNORECASE)
    
    def _load_lexicon(self):
        """加载饮食模式对齐资源"""
        if not os.path.exists(self.lexicon_file):
            logger.warning(f"DietaryPattern 词表不存在，回退为规则规范化: {self.lexicon_file}")
            return
        
        try:
            with open(self.lexicon_file, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
            
            for concept in self.lexicon:
                for alias in concept.get("all_aliases", []):
                    norm = self._normalize_lookup_text(alias)
                    if norm:
                        self.alias_index[norm] = {
                            "concept": concept,
                            "matched_alias": alias
                        }
            
            logger.info(f"DietaryPattern 词表加载完成，共 {len(self.lexicon)} 个概念")
        except Exception as e:
            logger.warning(f"加载 DietaryPattern 词表失败，回退为规则规范化: {e}")
            self.lexicon = []
            self.alias_index = {}
    
    def _lookup_lexicon(self, entity: str) -> Optional[Dict[str, Any]]:
        """在词表中执行保守查找"""
        variants = [
            entity,
            self._remove_parenthetical(entity),
            self._singularize_terminal_diets(entity),
            self._singularize_terminal_diets(self._remove_parenthetical(entity)),
        ]
        seen = set()
        for variant in variants:
            norm = self._normalize_lookup_text(variant)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            if norm in self.alias_index:
                hit = self.alias_index[norm]
                concept = hit["concept"]
                return {
                    "concept": concept,
                    "matched_alias": hit["matched_alias"],
                    "matched_variant": variant
                }
        return None
        
    def normalize(self, entity: str, abstract: str = "") -> AlignmentResult:
        """规范化饮食模式实体
        
        Args:
            entity: 待规范化的饮食模式实体
            abstract: 原始摘要（用于记录）
            
        Returns:
            AlignmentResult: 规范化结果（状态为NEW_CONCEPT）
        """
        original = entity
        
        # 1. 展开缩写
        for abbr, full in self.abbreviation_map.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            entity = re.sub(pattern, full, entity, flags=re.IGNORECASE)
        
        # 2. 去除冠词
        words = entity.split()
        words = [w for w in words if w.lower() not in self.articles]
        entity = ' '.join(words)
        
        # 3. 词表优先：仅接受 exact/synonym 级别命中
        lexicon_hit = self._lookup_lexicon(entity)
        if lexicon_hit is not None:
            concept = lexicon_hit["concept"]
            return AlignmentResult(
                original_entity=original,
                entity_type="DietaryPattern",
                status=AlignmentStatus.DIRECT_HIT,
                standard_id=concept.get("preferred_source_id"),
                standard_name=concept.get("canonical_label"),
                similarity=1.0,
                synonyms=concept.get("all_aliases", []),
                attributes={
                    "concept_id": concept.get("concept_id"),
                    "preferred_source": concept.get("preferred_source"),
                    "preferred_source_url": concept.get("preferred_source_url"),
                    "curation_bucket": concept.get("curation_bucket"),
                    "matched_alias": lexicon_hit.get("matched_alias"),
                    "matched_variant": lexicon_hit.get("matched_variant"),
                    "sources": concept.get("sources", [])
                },
                is_valid=True,
                is_new_concept=False,
                description=f"Matched via curated dietary pattern lexicon: {lexicon_hit.get('matched_alias')}",
                reason="Exact/synonym hit in verified DietaryPattern lexicon"
            )
        
        # 4. Title Case
        entity = entity.title()
        
        # 5. 去除多余空格
        entity = re.sub(r'\s+', ' ', entity).strip()
        
        # 若无可靠标准命中，则按开放世界策略保留为新概念
        return AlignmentResult(
            original_entity=original,
            entity_type="DietaryPattern",
            status=AlignmentStatus.NEW_CONCEPT,  # 作为新概念纳入
            standard_name=entity,
            similarity=1.0,
            is_valid=True,
            is_new_concept=True,
            description=f"No verified lexicon hit; normalized dietary pattern from: {original}",
            reason="No exact/synonym hit in verified DietaryPattern lexicon; retained as new concept"
        )


class GroundingAgent:
    """Grounding Agent（语义锚定智能体）"""
    
    def __init__(self, use_cache: bool = True):
        """初始化Grounding Agent
        
        Args:
            use_cache: 是否使用预编码缓存（默认True）
        """
        self.disease_aligner = DiseaseAligner()
        
        if use_cache and self.disease_aligner.load_from_cache():
            logger.info("使用预编码的疾病嵌入缓存")
        else:
            self.do_loader = DiseaseOntologyLoader(HUMANIDO_FILE)
            self.do_loader.load()
            self.disease_aligner.do_loader = self.do_loader
        
        self.food_nutrient_aligner = FoodNutrientAligner(INDEX_DIR)
        self.food_nutrient_aligner.load()
        
        self.dietary_normalizer = DietaryPatternNormalizer()
        
        self.feedback_requests = []
        
        logger.info("=" * 70)
        logger.info("Grounding Agent（语义锚定智能体）初始化完成")
        logger.info("=" * 70)
    
    def load_input_data(self):
        """加载输入数据
        
        Returns:
            tuple: (df_triples, dict_full)
        """
        # 加载三元组（Excel）
        df_triples = pd.read_excel(QUALITY_DECISION_LATEST_XLSX)
        logger.info(f"成功加载三元组数据: {len(df_triples)} 条")
        
        # 加载完整数据（JSON，含摘要）
        dict_full = {}
        if os.path.exists(QUALITY_DECISION_LATEST_JSON):
            with open(QUALITY_DECISION_LATEST_JSON, 'r', encoding='utf-8') as f:
                dict_full = json.load(f)
            logger.info(f"成功加载完整数据: {len(dict_full)} 条")
        
        return df_triples, dict_full
    
    def _get_entity_type(self, subject: str, obj: str, source_cluster: str, 
                         subject_type: Optional[str] = None, object_type: Optional[str] = None) -> Tuple[str, str, str]:
        """根据主题簇确定实体类型，优先使用NER识别的类型
        
        Returns:
            Tuple: (head_type, relation, tail_type)
        """
        from settings import TOPIC_CONFIG
        
        # 优先使用NER识别的实体类型
        head_type = subject_type
        tail_type = object_type
        
        # 如果NER没有识别出类型，则根据主题配置确定
        if not head_type and source_cluster in TOPIC_CONFIG:
            head_type = TOPIC_CONFIG[source_cluster]['head_type']
        if not tail_type and source_cluster in TOPIC_CONFIG:
            tail_type = TOPIC_CONFIG[source_cluster]['tail_type']
        
        # 默认类型推断
        if not head_type:
            head_type = "Unknown"
        if not tail_type:
            tail_type = "Unknown"
        
        return (head_type, "", tail_type)
    
    def ground_single_triple(self, triple: Dict, abstract: str = "", is_retry: bool = False) -> Dict:
        """对齐单个三元组
        
        Args:
            triple: 三元组字典
            abstract: 原始摘要
            is_retry: 是否为重试
            
        Returns:
            Dict: 对齐后的三元组
        """
        result = triple.copy()
        
        # 获取实体类型，优先使用NER识别的类型
        source_cluster = triple.get('source_cluster', triple.get('topic', 'T1'))
        subject_type = triple.get('subject_type', None)
        object_type = triple.get('object_type', None)
        head_type, _, tail_type = self._get_entity_type(
            triple.get('subject', ''),
            triple.get('object', ''),
            source_cluster,
            subject_type,
            object_type
        )
        
        # 对齐头实体
        head_entity = triple.get('subject', '')
        head_alignment = self._align_entity(head_entity, head_type, abstract, is_retry)
        head_align_dict = asdict(head_alignment)
        head_align_dict['status'] = head_alignment.status.value
        head_align_dict['similarity'] = float(head_align_dict['similarity'])
        result['head_alignment'] = head_align_dict
        
        # 对齐尾实体
        tail_entity = triple.get('object', '')
        tail_alignment = self._align_entity(tail_entity, tail_type, abstract, is_retry)
        tail_align_dict = asdict(tail_alignment)
        tail_align_dict['status'] = tail_alignment.status.value
        tail_align_dict['similarity'] = float(tail_align_dict['similarity'])
        result['tail_alignment'] = tail_align_dict
        
        # 检查是否需要反馈
        if (head_alignment.status == AlignmentStatus.FEEDBACK_REQUESTED or 
            tail_alignment.status == AlignmentStatus.FEEDBACK_REQUESTED):
            result['needs_feedback'] = True
            if head_alignment.status == AlignmentStatus.FEEDBACK_REQUESTED:
                result['feedback_entity'] = 'head'
                result['feedback_entity_text'] = head_entity
            else:
                result['feedback_entity'] = 'tail'
                result['feedback_entity_text'] = tail_entity
        else:
            result['needs_feedback'] = False
        
        return result
    
    def _align_entity(self, entity: str, entity_type: str, abstract: str = "", is_retry: bool = False) -> AlignmentResult:
        """对齐单个实体
        
        Args:
            entity: 待对齐实体
            entity_type: 实体类型
            abstract: 原始摘要（用于开放世界验证）
            is_retry: 是否为重试
            
        Returns:
            AlignmentResult: 对齐结果
        """
        if not entity:
            return AlignmentResult(
                original_entity=entity,
                entity_type=entity_type,
                status=AlignmentStatus.OOV
            )
        
        if entity_type == "Disease":
            return self.disease_aligner.align(entity, abstract, is_retry)
        elif entity_type == "Food":
            return self.food_nutrient_aligner.align_food(entity, abstract)
        elif entity_type == "Nutrient":
            return self.food_nutrient_aligner.align_nutrient(entity, abstract)
        elif entity_type == "DietaryPattern":
            return self.dietary_normalizer.normalize(entity, abstract)
        else:
            return AlignmentResult(
                original_entity=entity,
                entity_type=entity_type,
                status=AlignmentStatus.OOV
            )
    
    def run(self, max_records=None):
        """运行Grounding Agent
        
        Args:
            max_records (int, optional): 最大处理记录数. Defaults to None.
        """
        logger.info("开始运行Grounding Agent（语义锚定智能体）")
        
        # 加载输入数据
        df_triples, dict_full = self.load_input_data()
        
        # 限制处理记录数量
        if max_records:
            df_triples = df_triples.head(max_records)
            logger.info(f"限制处理前 {max_records} 条记录")
        
        # 处理每个三元组
        aligned_triples = []
        feedback_requests = []
        
        for idx, row in tqdm(df_triples.iterrows(), total=len(df_triples), desc="对齐三元组"):
            triple = row.to_dict()
            
            # 获取摘要
            pmid = str(triple.get('pmid', ''))
            abstract = ""
            if pmid in dict_full:
                pmid_data = dict_full[pmid]
                if 'final_triples' in pmid_data and len(pmid_data['final_triples']) > 0:
                    abstract = pmid_data['final_triples'][0].get('abstract', '')
            
            # 对齐
            aligned_triple = self.ground_single_triple(triple, abstract)
            aligned_triples.append(aligned_triple)
            
            # 收集反馈请求
            if aligned_triple.get('needs_feedback', False):
                head_align = aligned_triple.get('head_alignment', {})
                tail_align = aligned_triple.get('tail_alignment', {})
                
                feedback_entity = aligned_triple.get('feedback_entity', '')
                if feedback_entity == 'head':
                    feedback_entity_text = head_align.get('original_entity', '')
                    candidate_name = head_align.get('standard_name', '')
                else:
                    feedback_entity_text = tail_align.get('original_entity', '')
                    candidate_name = tail_align.get('standard_name', '')
                
                feedback_requests.append({
                    'pmid': pmid,
                    'triple_idx': idx,
                    'original_triple': aligned_triple,
                    'feedback_entity': feedback_entity,
                    'feedback_entity_text': feedback_entity_text,
                    'candidate_name': candidate_name,
                    'abstract': abstract,
                    'is_retry': False
                })
        
        logger.info(f"对齐完成: {len(aligned_triples)} 个三元组")
        logger.info(f"反馈请求: {len(feedback_requests)} 个")
        
        # 保存结果
        self.save_results(aligned_triples, feedback_requests)
        
        logger.info("Grounding Agent运行完成")
    
    def save_results(self, aligned_triples: List[Dict], feedback_requests: List[Dict]):
        """保存结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存反馈请求到单独文件（用于反馈回路②）
        if feedback_requests:
            feedback_data = {
                'feedback_requests': feedback_requests,
                'timestamp': timestamp
            }
            with open(FEEDBACK_REFINEMENT_REQUESTS, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(feedback_requests)} 个反馈请求到 {FEEDBACK_REFINEMENT_REQUESTS}")
        else:
            # 如果没有反馈请求，删除旧的反馈请求文件
            if os.path.exists(FEEDBACK_REFINEMENT_REQUESTS):
                os.remove(FEEDBACK_REFINEMENT_REQUESTS)
                logger.info("无反馈请求，已删除旧的反馈请求文件")
        
        # 2. 保存为JSON（完整结果）
        json_result = {
            'aligned_triples': aligned_triples,
            'feedback_requests': feedback_requests,
            'timestamp': timestamp
        }
        
        with open(GROUNDING_ALIGNMENT_LATEST_JSON, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        json_archive = os.path.join(GROUNDING_ALIGNMENT_DIR, f"grounding_alignment_{timestamp}.json")
        with open(json_archive, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        # 2. 保存为Excel（简化版，用于查看）
        excel_rows = []
        for triple in aligned_triples:
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
                'needs_feedback': triple.get('needs_feedback', False)
            }
            excel_rows.append(row)
        
        df_excel = pd.DataFrame(excel_rows)
        df_excel.to_excel(GROUNDING_ALIGNMENT_LATEST, index=False)
        
        excel_archive = os.path.join(GROUNDING_ALIGNMENT_DIR, f"grounding_alignment_{timestamp}.xlsx")
        df_excel.to_excel(excel_archive, index=False)
        
        logger.info(f"已保存结果:")
        logger.info(f"  - JSON: {GROUNDING_ALIGNMENT_LATEST_JSON}")
        logger.info(f"  - Excel: {GROUNDING_ALIGNMENT_LATEST}")
        
        # 统计
        status_counts = {}
        for triple in aligned_triples:
            for key in ['head_alignment', 'tail_alignment']:
                status = triple.get(key, {}).get('status', '')
                status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info("=" * 60)
        logger.info("对齐统计")
        logger.info("=" * 60)
        for status, count in sorted(status_counts.items()):
            logger.info(f"  {status}: {count}")
        logger.info(f"  反馈请求: {len(feedback_requests)}")
        logger.info("=" * 60)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Grounding Agent: 语义锚定智能体')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量（用于批量运行控制）')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Grounding Agent（语义锚定智能体）启动")
    logger.info("分层异构对齐策略 + LLM语义验证 + 反馈回路②")
    if args.limit:
        logger.info(f"限制处理数量: {args.limit}")
    logger.info("=" * 70)
    
    agent = GroundingAgent()
    agent.run(max_records=args.limit)
    
    logger.info("=" * 70)
    logger.info("Grounding Agent处理完成")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()


