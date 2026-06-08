#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_entity.py - 预训练模型：生成种子实体

功能：
- 读取semantic_triage的输出结果
- 根据命中的主题簇，调用相应的预训练模型
- 生成种子实体和模型置信度
- 输出带时间戳的归档文件和固定名latest文件
- 支持断点续跑

按照DIETA框架实现 - 种子实体生成（预处理步骤）

输入：
- results/semantic_triage/semantic_triage_latest.xlsx

输出：
- results/seed_entity/seed_entity_latest.json（固定名，每次覆盖）
- results/seed_entity/seed_entity_[时间戳].json（带时间戳归档）
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import signal
import json
import pandas as pd
from tqdm import tqdm
from settings import (
    DISEASE_MODEL_NAME, DISEASE_MODEL_AGGREGATION, DISEASE_MIN_CONFIDENCE,
    NUTRIENT_MODEL_NAME, NUTRIENT_MODEL_AGGREGATION, NUTRIENT_MIN_CONFIDENCE,
    FOOD_MODEL_NAME, FOOD_MODEL_AGGREGATION, FOOD_MIN_CONFIDENCE,
    DIETARY_PATTERN_MODEL_NAME, ENTITY_MIN_LENGTH,
    SEMANTIC_TRIAGE_LATEST, SEED_ENTITY_DIR, SEED_ENTITY_LATEST,
    TOPIC_CONFIG
)
from utils import setup_logging, load_processed_ids, load_existing_results, save_results_with_timestamp

logger = setup_logging(SEED_ENTITY_DIR, 'seed_entity.log')

os.makedirs(SEED_ENTITY_DIR, exist_ok=True)


class SeedEntityGenerator:
    """种子实体生成器"""
    
    MODEL_CONFIGS = {
        'Disease': {
            'name': DISEASE_MODEL_NAME,
            'aggregation': DISEASE_MODEL_AGGREGATION,
            'min_confidence': DISEASE_MIN_CONFIDENCE
        },
        'Nutrient': {
            'name': NUTRIENT_MODEL_NAME,
            'aggregation': NUTRIENT_MODEL_AGGREGATION,
            'min_confidence': NUTRIENT_MIN_CONFIDENCE
        },
        'Food': {
            'name': FOOD_MODEL_NAME,
            'aggregation': FOOD_MODEL_AGGREGATION,
            'min_confidence': FOOD_MIN_CONFIDENCE
        }
    }
    
    def __init__(self):
        """初始化种子实体生成器"""
        self.models = {}
        self._load_models()
        self.running = True
        self.results = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.info(f"收到信号 {signum}，准备保存进度并退出")
        self.running = False
    
    def _load_single_model(self, model_type: str):
        """加载单个预训练模型"""
        config = self.MODEL_CONFIGS[model_type]
        try:
            logger.info(f"加载{model_type}模型: {config['name']}")
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            tokenizer = AutoTokenizer.from_pretrained(config['name'])
            model = AutoModelForTokenClassification.from_pretrained(config['name'])
            self.models[model_type] = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy=config['aggregation']
            )
            logger.info(f"{model_type}模型加载成功")
        except Exception as e:
            logger.error(f"加载{model_type}模型失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise RuntimeError(f"{model_type}模型加载失败，必须使用预训练模型") from e
    
    def _load_models(self):
        """加载所有预训练模型"""
        self.models = {'Disease': None, 'Nutrient': None, 'Food': None}
        
        for model_type in ['Disease', 'Nutrient', 'Food']:
            self._load_single_model(model_type)
        
        logger.info("DietaryPattern：无预训练模型，实体识别将由LLM直接进行")
        
        loaded_models = [k for k, v in self.models.items() if v is not None]
        if len(loaded_models) != 3:
            raise RuntimeError(f"部分模型加载失败，仅加载了: {loaded_models}，必须使用预训练模型")
    
    def _normalize_model_output(self, raw_output: list, model_type: str):
        """标准化模型输出"""
        config = self.MODEL_CONFIGS[model_type]
        normalized = []
        for item in raw_output:
            entity = item['word'].strip()
            confidence = float(item['score'])
            if confidence >= config['min_confidence'] and len(entity) >= ENTITY_MIN_LENGTH:
                normalized.append({
                    "entity": entity,
                    "type": model_type,
                    "confidence": confidence
                })
        return normalized
    
    def load_input_data(self):
        """加载输入数据"""
        try:
            df = pd.read_excel(SEMANTIC_TRIAGE_LATEST)
            df_filtered = df[df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6']].sum(axis=1) > 0]
            logger.info(f"成功加载输入数据，共 {len(df_filtered)} 条有效记录")
            return df_filtered
        except Exception as e:
            logger.error(f"加载输入数据失败: {e}")
            raise
    
    def extract_entities(self, abstract: str, model_types: list):
        """提取实体"""
        entities = {
            "Food": [],
            "Disease": [],
            "Nutrient": [],
            "DietaryPattern": []
        }
        model_confidence = {
            "Food": [],
            "Disease": [],
            "Nutrient": []
        }
        
        logger.info(f"处理摘要: {abstract[:100]}...")
        logger.info(f"需要使用的模型类型: {model_types}")
        
        for model_type in model_types:
            if model_type not in self.models or self.models[model_type] is None:
                raise RuntimeError(f"{model_type}模型未加载，必须使用预训练模型")
            
            try:
                logger.info(f"开始提取{model_type}实体")
                raw_output = self.models[model_type](abstract)
                logger.info(f"{model_type}模型原始输出: {raw_output}")
                normalized = self._normalize_model_output(raw_output, model_type)
                logger.info(f"{model_type}模型标准化输出: {normalized}")
                entities[model_type] = [item["entity"] for item in normalized]
                model_confidence[model_type] = normalized
                logger.info(f"提取到的{model_type}实体: {entities[model_type]}")
            except Exception as e:
                logger.error(f"提取{model_type}实体失败: {e}")
                raise RuntimeError(f"{model_type}实体提取失败，必须使用预训练模型") from e
        
        logger.info(f"最终实体提取结果: {entities}")
        return entities, model_confidence
    
    def process_single_record(self, row):
        """处理单条记录"""
        try:
            pmid = str(row['pmid'])
            abstract = str(row['abstract'])
            
            matched_clusters = []
            if pd.notna(row['matched_clusters']):
                matched_clusters = [t.strip() for t in str(row['matched_clusters']).split(',') if t.strip()]
            
            model_types = set()
            for topic in matched_clusters:
                if topic in TOPIC_CONFIG:
                    model_types.update(TOPIC_CONFIG[topic]['need_models'])
            
            seed_entities, model_confidence = self.extract_entities(abstract, list(model_types))
            
            return pmid, {
                "pmid": pmid,
                "abstract": abstract,
                "matched_clusters": matched_clusters,
                "seed_entities": seed_entities,
                "model_confidence": model_confidence,
                "processed": True,
                "error": None
            }
        except Exception as e:
            pmid = str(row['pmid'])
            logger.error(f"处理记录 {pmid} 失败: {e}")
            return pmid, {
                "pmid": pmid,
                "abstract": str(row['abstract']),
                "matched_clusters": [],
                "seed_entities": {
                    "Food": [], "Disease": [], "Nutrient": [], "DietaryPattern": []
                },
                "model_confidence": {"Food": [], "Disease": [], "Nutrient": []},
                "processed": False,
                "error": str(e)
            }
    
    def save_results(self, results):
        """保存结果"""
        save_results_with_timestamp(
            data=results,
            latest_file=SEED_ENTITY_LATEST,
            archive_dir=SEED_ENTITY_DIR,
            file_prefix='seed_entity'
        )
        logger.info(f"已保存 {len(results)} 条记录")
    
    def run(self, max_records=None):
        """运行种子实体生成器"""
        logger.info("开始运行种子实体生成器")
        
        df_input = self.load_input_data()
        processed_pmids = load_processed_ids(SEED_ENTITY_LATEST)
        
        df_to_process = df_input[~df_input['pmid'].astype(str).isin(processed_pmids)]
        
        if max_records:
            df_to_process = df_to_process.head(max_records)
            logger.info(f"限制处理前 {max_records} 条记录（快速实验模式）")
        
        logger.info(f"待处理 {len(df_to_process)} 条记录")
        
        self.results = {}
        existing = load_existing_results(SEED_ENTITY_LATEST)
        if existing:
            self.results = existing
            logger.info(f"加载了 {len(self.results)} 条已处理记录")
        
        processed_count = 0
        for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="生成种子实体"):
            if not self.running:
                logger.info("收到中止信号，停止处理")
                break
            
            pmid, result = self.process_single_record(row)
            self.results[pmid] = result
            processed_count += 1
            
            self.save_results(self.results)
            logger.info(f"已处理 {processed_count} 条记录，共 {len(df_to_process)} 条")
        
        if self.results:
            self.save_results(self.results)
        
        if not self.running:
            logger.info("种子实体生成器被中止")
        else:
            logger.info("种子实体生成器运行完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Seed Entity Generator: 种子实体生成器')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量（用于批量运行控制）')
    args = parser.parse_args()
    
    generator = SeedEntityGenerator()
    generator.run(max_records=args.limit)


