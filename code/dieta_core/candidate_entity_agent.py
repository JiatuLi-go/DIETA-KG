#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
candidate_entity_extraction.py - Extraction Agent（NER部分）：LLM多提示词实体识别

功能：
- 读取seed_entity的输出结果
- 使用多个提示词进行实体识别
- 实现自一致性投票
- 输出带时间戳的归档文件和固定名latest文件
- 支持断点续跑

按照DIETA框架实现 - Extraction Agent（NER部分）

输入：
- results/seed_entity/seed_entity_latest.json

输出：
- results/candidate_extraction/candidate_entity_extraction_latest.json（固定名，每次覆盖）
- results/candidate_extraction/candidate_entity_extraction_[时间戳].json（带时间戳归档）
- results/candidate_extraction/candidate_entity_extraction_latest.xlsx（Excel展平版，供人工查看）
"""

import os
import time
import json
import re
import pandas as pd
import requests
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_client import LLMClient
from settings import (
    API_KEY, API_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    API_RETRY_TIMES, API_RETRY_INTERVAL, BATCH_SIZE, NER_PROMPT_COUNT, TAU,
    SEMANTIC_TRIAGE_LATEST, SEED_ENTITY_LATEST, CANDIDATE_EXTRACTION_DIR, CANDIDATE_EXTRACTION_NER_LATEST_JSON, CANDIDATE_EXTRACTION_NER_LATEST_XLSX,
    TOPIC_CONFIG, PROMPTS_DIR, USE_PARALLEL, MAX_WORKERS, TOPIC_MAX_WORKERS
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CANDIDATE_EXTRACTION_DIR, 'candidate_entity_extraction.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保结果目录存在
os.makedirs(CANDIDATE_EXTRACTION_DIR, exist_ok=True)

class NERAgent:
    """Extraction Agent（NER部分）：LLM多提示词实体识别"""
    
    def __init__(self):
        """初始化Extraction Agent（NER部分）"""
        # 加载提示词
        self.llm_client = LLMClient()
        self.prompts = self._load_prompts()
        # 初始化缓存
        self.llm_cache = {}  # 缓存LLM调用结果
        self.prompt_cache = {}  # 缓存提示词构建结果
        self.parsed_response_cache = {}  # 缓存解析后的结果
        # 预编译正则表达式，提高解析效率
        self.re_code_block = re.compile(r'```json\n([\s\S]*?)\n```', re.DOTALL)
        self.re_json_array = re.compile(r'\[\s*\{[\s\S]*\}\s*\]', re.DOTALL)
        self.re_json_object = re.compile(r'\{[\s\S]*?abstractID[\s\S]*?\}', re.DOTALL)
        self.re_trailing_comma_array = re.compile(r',\s*(\]|\})', re.DOTALL)
        self.re_trailing_comma_object = re.compile(r',\s*(\})', re.DOTALL)
        self.re_truncated_str1 = re.compile(r'"\.\.\."')
        self.re_truncated_str2 = re.compile(r'\.\.\."')
        self.re_truncated_str3 = re.compile(r'"\.\.\.')
        self.re_entity_normalize = re.compile(r'##\w+')
        # 初始化性能监控（详细版）
        self.performance = {
            'total_time': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'batch_processing_time': 0,
            'parsing_time': 0,
            'llm_time': 0,
            'save_time': 0,
            'prompt_build_time': 0,
            'vote_calculation_time': 0,
            'data_loading_time': 0
        }
    
    def _load_prompts(self):
        """加载提示词
        
        Returns:
            dict: 提示词字典
        """
        prompts = {}
        for topic in TOPIC_CONFIG.keys():
            prompts[topic] = {}
            for i in range(1, NER_PROMPT_COUNT + 1):
                prompt_file = os.path.join(PROMPTS_DIR, f"ner_{topic}_p{i}.txt")
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompts[topic][f"p{i}"] = f.read().strip()
                else:
                    logger.warning(f"提示词文件不存在: {prompt_file}")
        return prompts
    
    def load_input_data(self):
        """加载输入数据
        
        Returns:
            dict: 输入数据
        """
        start_time = time.time()
        try:
            with open(SEED_ENTITY_LATEST, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.performance['data_loading_time'] = time.time() - start_time
            logger.info(f"成功加载输入数据，共 {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"加载输入数据失败: {e}")
            raise
    
    def load_processed_pmids(self):
        """加载已处理的pmid
        
        Returns:
            set: 已处理的pmid集合
        """
        processed_pmids = set()
        if os.path.exists(CANDIDATE_EXTRACTION_NER_LATEST_JSON):
            try:
                with open(CANDIDATE_EXTRACTION_NER_LATEST_JSON, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    processed_pmids = set(data.keys())
                logger.info(f"已处理 {len(processed_pmids)} 条记录")
            except Exception as e:
                logger.warning(f"读取已处理记录失败: {e}")
        return processed_pmids
    
    def build_batch_prompt(self, topic, batch_data, prompt_key):
        """构建批处理提示词
        
        Args:
            topic (str): 主题
            batch_data (list): 批处理数据
            prompt_key (str): 提示词键
            
        Returns:
            str: 完整的提示词
        """
        start_time = time.time()
        
        # 生成缓存键
        cache_key = f"{topic}:{prompt_key}:{hash(str(batch_data))}"
        
        # 检查缓存
        if cache_key in self.prompt_cache:
            logger.info(f"使用缓存的提示词: {cache_key}")
            self.performance['prompt_build_time'] += time.time() - start_time
            return self.prompt_cache[cache_key]
        
        # 构建Batch Data部分
        batch_str = "Batch Data:\n[\n"
        for item in batch_data:
            # 处理DietaryPattern为空的情况
            seed_entities = item['seed_entities']
            if not seed_entities.get('DietaryPattern'):
                seed_entities['DietaryPattern'] = []
                # 添加注释
                batch_str += '  {\n'
                batch_str += '    "abstractID": "' + str(item['abstractID']) + '",\n'
                batch_str += '    "abstract": "' + str(item['abstract']) + '",\n'
                batch_str += '    "seed_entities": {\n'
                batch_str += '      "Food": ["' + '", "'.join([str(e) for e in seed_entities['Food']]) + '"],\n'
                batch_str += '      "Disease": ["' + '", "'.join([str(e) for e in seed_entities['Disease']]) + '"],\n'
                batch_str += '      "Nutrient": ["' + '", "'.join([str(e) for e in seed_entities['Nutrient']]) + '"],\n'
                batch_str += '      "DietaryPattern": [] // No pretrained model available for Dietary Pattern - please identify from text directly\n'
                batch_str += '    }\n'
                batch_str += '  },\n'
            else:
                batch_str += '  {\n'
                batch_str += '    "abstractID": "' + str(item['abstractID']) + '",\n'
                batch_str += '    "abstract": "' + str(item['abstract']) + '",\n'
                batch_str += '    "seed_entities": {\n'
                batch_str += '      "Food": ["' + '", "'.join([str(e) for e in seed_entities['Food']]) + '"],\n'
                batch_str += '      "Disease": ["' + '", "'.join([str(e) for e in seed_entities['Disease']]) + '"],\n'
                batch_str += '      "Nutrient": ["' + '", "'.join([str(e) for e in seed_entities['Nutrient']]) + '"],\n'
                batch_str += '      "DietaryPattern": ["' + '", "'.join([str(e) for e in seed_entities['DietaryPattern']]) + '"]\n'
                batch_str += '    }\n'
                batch_str += '  },\n'
        # 移除最后一个逗号
        if batch_data:
            batch_str = batch_str.rstrip(",\n") + "\n"
        batch_str += "]\n\n"
        
        # Feedback channel: optional reflection reports are injected only
        # when the Feedback Controller explicitly requests re-extraction.
        reflection_blocks = []
        for item in batch_data:
            if item.get('reflection_report'):
                reflection_blocks.append(
                    f"PMID/abstractID: {item.get('abstractID', '')}\n"
                    f"{item.get('reflection_report', '')}"
                )
        if reflection_blocks:
            batch_str += "\n[Feedback / Reflection Guidance]\n"
            batch_str += "\n\n".join(reflection_blocks)[:4000]
            batch_str += "\nPlease revise entity recognition under the same entity schema and output JSON format.\n\n"

        # 添加固定指令部分
        prompt_content = self.prompts.get(topic, {}).get(prompt_key, "")
        batch_str += prompt_content
        
        # 缓存提示词
        self.prompt_cache[cache_key] = batch_str
        
        self.performance['prompt_build_time'] += time.time() - start_time
        return batch_str
    
    def call_llm(self, prompt):
        """Call the LLM through the centralized DIETA LLM gateway.

        The public method and returned string/None contract are preserved so that
        downstream JSON parsing and output file schemas remain unchanged.
        """
        cache_key = hash(prompt)
        if cache_key in self.llm_cache:
            logger.info("使用缓存的LLM结果")
            self.performance['cache_hits'] += 1
            return self.llm_cache[cache_key]

        self.performance['llm_calls'] += 1
        llm_start_time = time.time()
        try:
            content = self.llm_client.generate(
                prompt=prompt,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                use_cache=True,
            ).strip()
            self.llm_cache[cache_key] = content
            self.performance['llm_time'] += time.time() - llm_start_time
            return content
        except Exception as e:
            logger.error(f"LLM gateway call failed: {e}")
            self.performance['llm_time'] += time.time() - llm_start_time
            return None
    
    def repair_json(self, json_str):
        """尝试修复损坏的JSON字符串
        
        Args:
            json_str (str): 损坏的JSON字符串
            
        Returns:
            str: 修复后的JSON字符串
        """
        import re
        
        # 复制原始字符串
        repaired = json_str
        
        # 1. 修复末尾缺少闭合括号的情况
        # 统计括号数量
        open_brackets = repaired.count('[')
        close_brackets = repaired.count(']')
        open_braces = repaired.count('{')
        close_braces = repaired.count('}')
        
        # 补充缺失的闭合括号
        while close_brackets < open_brackets:
            repaired = repaired.rstrip() + ']'
            close_brackets += 1
        
        # 补充缺失的闭合大括号
        while close_braces < open_braces:
            repaired = repaired.rstrip() + '}'
            close_braces += 1
        
        # 2. 修复数组中最后一个元素后面的逗号
        repaired = re.sub(r',\s*(\]|\})', r'\1', repaired)
        
        # 3. 修复对象中最后一个属性后面的逗号
        repaired = re.sub(r',\s*(\})', r'\1', repaired)
        
        # 4. 修复被截断的字符串（处理...结尾的情况）
        repaired = re.sub(r'"\.\.\."', '""', repaired)
        repaired = re.sub(r'\.\.\."', '"', repaired)
        repaired = re.sub(r'"\.\.\.', '"', repaired)
        
        # 5. 修复未闭合的字符串
        # 这是一个简化的处理，实际情况可能更复杂
        # 统计引号数量，如果是奇数，尝试在末尾添加一个引号
        quote_count = repaired.count('"')
        if quote_count % 2 != 0:
            # 找到最后一个引号的位置
            last_quote_pos = repaired.rfind('"')
            if last_quote_pos != -1:
                # 检查是否在字符串内部（简化处理）
                repaired = repaired[:last_quote_pos + 1]
        
        return repaired
    
    def parse_llm_response(self, response, batch_data):
        """解析LLM返回结果
        
        Args:
            response (str): LLM返回结果
            batch_data (list): 批处理数据
            
        Returns:
            dict: 解析后的实体结果
        """
        start_time = time.time()
        results = {}
        if not response:
            # 如果返回结果为空，所有实体为空列表
            for item in batch_data:
                results[item['abstractID']] = []
            self.performance['parsing_time'] += time.time() - start_time
            return results
        
        # 尝试解析JSON格式的返回结果
        try:
            import re
            
            # 1. 处理Markdown代码块（Qwen2.5-72B格式）
            code_block_match = re.search(r'```json\n([\s\S]*?)\n```', response)
            if code_block_match:
                response = code_block_match.group(1)
                logger.info("找到并处理Markdown代码块")
            
            # 2. 寻找完整的JSON数组（贪婪匹配，确保匹配到完整的数组）
            json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    entities_list = json.loads(json_str)
                    if isinstance(entities_list, list):
                        # 按abstractID组织结果
                        for item in entities_list:
                            abstract_id = item.get('abstractID')
                            # 处理不同的字段名
                            entities = item.get('entities', item.get('entityList', []))
                            # 标准化实体格式
                            normalized_entities = []
                            for entity in entities:
                                if isinstance(entity, dict):
                                    # 处理不同的字段名
                                    entity_text = entity.get('entity', entity.get('entityName', ''))
                                    entity_type = entity.get('type', entity.get('entityType', ''))
                                    if entity_text:
                                        normalized_entities.append({
                                            'entity': entity_text,
                                            'type': entity_type
                                        })
                            results[abstract_id] = normalized_entities
                except Exception as e:
                    logger.warning(f"解析JSON数组失败，尝试修复: {e}")
                    # 尝试修复JSON
                    repaired_json = self.repair_json(json_str)
                    try:
                        entities_list = json.loads(repaired_json)
                        if isinstance(entities_list, list):
                            # 按abstractID组织结果
                            for item in entities_list:
                                abstract_id = item.get('abstractID')
                                # 处理不同的字段名
                                entities = item.get('entities', item.get('entityList', []))
                                # 标准化实体格式
                                normalized_entities = []
                                for entity in entities:
                                    if isinstance(entity, dict):
                                        # 处理不同的字段名
                                        entity_text = entity.get('entity', entity.get('entityName', ''))
                                        entity_type = entity.get('type', entity.get('entityType', ''))
                                        if entity_text:
                                            normalized_entities.append({
                                                'entity': entity_text,
                                                'type': entity_type
                                            })
                                results[abstract_id] = normalized_entities
                        logger.info("JSON修复成功")
                    except Exception as e2:
                        logger.error(f"修复JSON后仍然失败: {e2}")
                        logger.error(f"原始JSON: {json_str[:300]}...")
                        logger.error(f"修复后JSON: {repaired_json[:300]}...")
            
            # 3. 处理单个JSON对象的情况
            if not results:
                # 尝试寻找单个JSON对象
                json_matches = re.findall(r'\{[\s\S]*?abstractID[\s\S]*?\}', response)
                for json_str in json_matches:
                    try:
                        entity_item = json.loads(json_str)
                        if isinstance(entity_item, dict):
                            abstract_id = entity_item.get('abstractID')
                            entities = entity_item.get('entities', [])
                            if abstract_id:
                                results[abstract_id] = entities
                    except Exception as e:
                        logger.warning(f"解析JSON对象失败，尝试修复: {e}")
                        # 尝试修复JSON
                        repaired_json = self.repair_json(json_str)
                        try:
                            entity_item = json.loads(repaired_json)
                            if isinstance(entity_item, dict):
                                abstract_id = entity_item.get('abstractID')
                                entities = entity_item.get('entities', [])
                                if abstract_id:
                                    results[abstract_id] = entities
                            logger.info("JSON对象修复成功")
                        except Exception as e2:
                            logger.error(f"修复JSON对象后仍然失败: {e2}")
                            logger.error(f"JSON字符串: {json_str[:300]}...")
            
            # 4. 确保所有batch_data中的abstractID都有结果
            for item in batch_data:
                if item['abstractID'] not in results:
                    results[item['abstractID']] = []
        except Exception as e:
            logger.error(f"解析LLM返回结果失败: {e}")
            # 解析失败时，所有实体为空列表
            for item in batch_data:
                results[item['abstractID']] = []
        
        self.performance['parsing_time'] += time.time() - start_time
        return results
    
    def normalize_entity(self, entity):
        """标准化实体
        
        Args:
            entity (str): 实体字符串
            
        Returns:
            str: 标准化后的实体
        """
        # 移除奇怪的标记（如##2、##ly等）
        import re
        entity = re.sub(r'##\w+', '', entity)
        # 大小写不敏感，连字符和空格等价
        return entity.lower().replace('-', ' ').strip()
    
    def validate_entity_type(self, entity_type):
        """验证实体类型
        
        Args:
            entity_type (str): 实体类型
            
        Returns:
            str: 验证后的实体类型
        """
        valid_types = ['Food', 'Disease', 'Nutrient', 'DietaryPattern']
        if entity_type in valid_types:
            return entity_type
        # 尝试修正常见错误
        type_mapping = {
            'food': 'Food',
            'disease': 'Disease',
            'nutrient': 'Nutrient',
            'dietarypattern': 'DietaryPattern',
            'dietary pattern': 'DietaryPattern'
        }
        return type_mapping.get(entity_type.lower(), 'Disease')
    
    def calculate_votes(self, all_entities):
        """计算实体投票
        
        Args:
            all_entities (dict): 所有提示词的实体结果
            
        Returns:
            dict: 实体投票结果
        """
        start_time = time.time()
        
        # 统计每个实体的出现次数
        entity_counts = {}
        for prompt_key, entities in all_entities.items():
            # 记录当前提示词中已经处理过的实体，避免重复计数
            processed_entities = set()
            for entity in entities:
                norm_entity = self.normalize_entity(entity['entity'])
                # 跳过空实体
                if not norm_entity:
                    continue
                # 验证实体类型
                valid_type = self.validate_entity_type(entity['type'])
                # 确保每个实体在每个提示词中只被计数一次
                if norm_entity not in processed_entities:
                    if norm_entity not in entity_counts:
                        entity_counts[norm_entity] = {
                            'original': entity['entity'],
                            'type': valid_type,
                            'count': 0,
                            'prompts': []
                        }
                    entity_counts[norm_entity]['count'] += 1
                    entity_counts[norm_entity]['prompts'].append(prompt_key)
                    processed_entities.add(norm_entity)
        
        # 计算置信度并过滤
        final_entities = []
        for norm_entity, info in entity_counts.items():
            confidence = info['count'] / NER_PROMPT_COUNT
            # 确保置信度在0-1之间
            confidence = min(1.0, max(0.0, confidence))
            if confidence >= TAU:
                final_entities.append({
                    "entity": info['original'],
                    "type": info['type'],
                    "confidence": confidence,
                    "votes": info['count']
                })
        
        self.performance['vote_calculation_time'] += time.time() - start_time
        return final_entities
    
    def process_single_record_topic(self, pmid, data, topic):
        """处理单条记录的单个主题
        
        Args:
            pmid (str): 文献ID
            data (dict): 记录数据
            topic (str): 主题
            
        Returns:
            dict: 处理结果
        """
        results = {}
        
        # 构建单条记录的数据
        single_data = [{
            "abstractID": pmid,
            "abstract": data['abstract'],
            "seed_entities": data['seed_entities'],
            "reflection_report": data.get('reflection_report', '')
        }]
        
        # 对每个提示词运行LLM
        all_entities = {}
        # 确保使用所有5个提示词
        for i in range(1, NER_PROMPT_COUNT + 1):
            prompt_key = f"p{i}"
            # 构建提示词
            prompt = self.build_batch_prompt(topic, single_data, prompt_key)
            
            # 调用LLM
            response = self.call_llm(prompt)
            
            # 解析结果
            prompt_results = self.parse_llm_response(response, single_data)
            all_entities[prompt_key] = prompt_results
        
        # 收集该摘要的所有实体
        abstract_entities = {}
        for prompt_key, prompt_results in all_entities.items():
            abstract_entities[prompt_key] = prompt_results.get(pmid, [])
        
        # 计算投票结果
        final_entities = self.calculate_votes(abstract_entities)
        
        results = {
            "entities": final_entities,
            "raw_votes": abstract_entities
        }
        
        return results
    
    def process_single_topic(self, pmid, data, topic):
        """处理单个主题
        
        Args:
            pmid (str): 文献ID
            data (dict): 记录数据
            topic (str): 主题
            
        Returns:
            tuple: (topic, result)
        """
        try:
            if topic in self.prompts:
                # 处理单条记录的单个主题
                result = self.process_single_record_topic(pmid, data, topic)
                return topic, result
        except Exception as e:
            logger.error(f"处理记录 {pmid} 的主题 {topic} 失败: {e}")
        return topic, None
    
    def process_single_record(self, pmid, data):
        """处理单条记录
        
        Args:
            pmid (str): 文献ID
            data (dict): 记录数据
            
        Returns:
            dict: 处理结果
        """
        try:
            ner_results = {}
            matched_topics = data['matched_clusters']
            
            if not matched_topics:
                return {
                    "pmid": pmid,
                    "abstract": data['abstract'],
                    "matched_clusters": matched_topics,
                    "ner_results": ner_results,
                    "processed": True,
                    "error": None
                }
            
            # 并行处理多个主题
            with ThreadPoolExecutor(max_workers=min(TOPIC_MAX_WORKERS, len(matched_topics))) as executor:
                # 提交所有主题处理任务
                future_to_topic = {
                    executor.submit(self.process_single_topic, pmid, data, topic): topic
                    for topic in matched_topics
                }
                
                # 收集结果
                for future in as_completed(future_to_topic):
                    topic, result = future.result()
                    if result:
                        ner_results[topic] = result
            
            result = {
                "pmid": pmid,
                "abstract": data['abstract'],
                "matched_clusters": matched_topics,
                "ner_results": ner_results,
                "processed": True,
                "error": None
            }
            
            return result
        except Exception as e:
            logger.error(f"处理记录 {pmid} 失败: {e}")
            return {
                "pmid": pmid,
                "abstract": data['abstract'],
                "matched_clusters": data.get('matched_clusters', []),
                "ner_results": {},
                "processed": False,
                "error": str(e)
            }
    
    def process_batch_topic(self, batch_data, topic):
        """处理一批记录的单个主题
        
        Args:
            batch_data (list): 批处理数据列表
            topic (str): 主题
            
        Returns:
            dict: 处理结果
        """
        # 对每个提示词运行LLM
        all_prompt_results = {}
        # 确保使用所有5个提示词
        for i in range(1, NER_PROMPT_COUNT + 1):
            prompt_key = f"p{i}"
            # 构建提示词
            prompt = self.build_batch_prompt(topic, batch_data, prompt_key)
            
            # 调用LLM
            response = self.call_llm(prompt)
            
            # 解析结果
            prompt_results = self.parse_llm_response(response, batch_data)
            all_prompt_results[prompt_key] = prompt_results
        
        # 为每条记录计算投票结果
        topic_results = {}
        for item in batch_data:
            pmid = item['abstractID']
            # 收集该摘要的所有实体
            abstract_entities = {}
            for prompt_key, prompt_results in all_prompt_results.items():
                abstract_entities[prompt_key] = prompt_results.get(pmid, [])
            
            # 计算投票结果
            final_entities = self.calculate_votes(abstract_entities)
            
            topic_results[pmid] = {
                "entities": final_entities,
                "raw_votes": abstract_entities
            }
        
        return topic_results
    
    def process_batch_records(self, batch_records):
        """处理一批记录
        
        Args:
            batch_records (dict): 批处理记录
            
        Returns:
            dict: 处理结果
        """
        batch_results = {}
        
        # 准备批处理数据
        batch_data = []
        for pmid, data in batch_records.items():
            batch_data.append({
                "abstractID": pmid,
                "abstract": data['abstract'],
                "seed_entities": data['seed_entities']
            })
        
        # 收集所有唯一的主题
        all_topics = set()
        for pmid, data in batch_records.items():
            all_topics.update(data.get('matched_clusters', []))
        
        # 处理每个主题
        topic_results = {}
        for topic in all_topics:
            if topic in self.prompts:
                try:
                    # 处理该主题的所有记录
                    results = self.process_batch_topic(batch_data, topic)
                    topic_results[topic] = results
                except Exception as e:
                    logger.error(f"处理主题 {topic} 失败: {e}")
        
        # 组织最终结果
        for pmid, data in batch_records.items():
            matched_topics = data.get('matched_clusters', [])
            ner_results = {}
            
            # 收集该记录的所有主题结果
            for topic in matched_topics:
                if topic in topic_results and pmid in topic_results[topic]:
                    ner_results[topic] = topic_results[topic][pmid]
            
            batch_results[pmid] = {
                "pmid": pmid,
                "abstract": data['abstract'],
                "matched_clusters": matched_topics,
                "ner_results": ner_results,
                "processed": True,
                "error": None
            }
        
        return batch_results
    
    def run(self, use_parallel=True, max_workers=3, batch_size=10, reprocess=False, max_records=None):
        """运行Extraction Agent（NER部分）
        
        Args:
            use_parallel (bool): 是否使用并行处理
            max_workers (int): 最大并发数
            batch_size (int): 批处理大小
            reprocess (bool): 是否重新处理已有的记录
            max_records (int, optional): 最大处理记录数. Defaults to None.
        """
        logger.info("开始运行Extraction Agent（NER部分）")
        start_time = time.time()
        
        # 加载输入数据
        input_data = self.load_input_data()
        
        if reprocess:
            # 重新处理所有记录
            to_process = input_data
            logger.info(f"重新处理所有 {len(to_process)} 条记录")
        else:
            # 加载已处理的pmid
            processed_pmids = self.load_processed_pmids()
            
            # 过滤出未处理的记录
            to_process = {pmid: data for pmid, data in input_data.items() if pmid not in processed_pmids}
            logger.info(f"待处理 {len(to_process)} 条记录")
        
        # 限制处理记录数量
        if max_records:
            to_process = dict(list(to_process.items())[:max_records])
            logger.info(f"限制处理前 {max_records} 条记录")
        
        # 处理记录
        results = {}
        
        # 如果有已处理的记录，先加载它们
        if os.path.exists(CANDIDATE_EXTRACTION_NER_LATEST_JSON):
            try:
                with open(CANDIDATE_EXTRACTION_NER_LATEST_JSON, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"加载了 {len(results)} 条已处理记录")
            except Exception as e:
                logger.warning(f"加载已处理记录失败: {e}")
        
        # 分组为批次
        batches = []
        batch = {}
        for pmid, data in to_process.items():
            batch[pmid] = data
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = {}
        if batch:
            batches.append(batch)
        
        logger.info(f"共分成 {len(batches)} 个批次，每批次最多 {batch_size} 条记录")
        
        batch_start_time = time.time()
        
        if use_parallel:
            logger.info(f"使用并行处理，最大并发数: {max_workers}")
            # 并行处理批次
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有批次任务
                future_to_batch = {
                    executor.submit(self.process_batch_records, batch): batch
                    for batch in batches
                }
                
                # 收集结果
                for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="并行处理批次"):
                    try:
                        batch_results = future.result()
                        results.update(batch_results)
                        
                        # 每处理一个批次保存一次
                        self.save_results(results)
                    except Exception as e:
                        logger.error(f"处理批次时出错: {e}")
        else:
            logger.info("使用串行处理")
            # 串行处理批次
            for batch in tqdm(batches, total=len(batches), desc="处理批次"):
                batch_results = self.process_batch_records(batch)
                results.update(batch_results)
                
                # 每处理一个批次保存一次
                self.save_results(results)
        
        # 记录批处理时间
        self.performance['batch_processing_time'] = time.time() - batch_start_time
        
        # 保存最终结果
        self.save_results(results)
        
        end_time = time.time()
        self.performance['total_time'] = end_time - start_time
        
        # 报告性能指标
        logger.info("性能报告:")
        logger.info(f"总处理时间: {self.performance['total_time']:.2f}秒")
        logger.info(f"数据加载时间: {self.performance['data_loading_time']:.2f}秒")
        logger.info(f"提示词构建时间: {self.performance['prompt_build_time']:.2f}秒")
        logger.info(f"LLM调用时间: {self.performance['llm_time']:.2f}秒")
        logger.info(f"JSON解析时间: {self.performance['parsing_time']:.2f}秒")
        logger.info(f"投票计算时间: {self.performance['vote_calculation_time']:.2f}秒")
        logger.info(f"结果保存时间: {self.performance['save_time']:.2f}秒")
        logger.info(f"批处理时间: {self.performance['batch_processing_time']:.2f}秒")
        logger.info(f"LLM调用次数: {self.performance['llm_calls']}")
        logger.info(f"缓存命中次数: {self.performance['cache_hits']}")
        logger.info(f"缓存命中率: {self.performance['cache_hits'] / (self.performance['llm_calls'] + self.performance['cache_hits']) * 100:.2f}%" if (self.performance['llm_calls'] + self.performance['cache_hits']) > 0 else "缓存命中率: 0%")
        logger.info(f"处理记录数: {len(results)}")
        logger.info(f"平均处理时间/记录: {self.performance['total_time'] / len(results):.2f}秒/条" if len(results) > 0 else "平均处理时间/记录: N/A")
        
        logger.info("Extraction Agent（NER部分）运行完成")
    
    def save_results(self, results):
        """保存结果
        
        Args:
            results (dict): 处理结果字典
        """
        start_time = time.time()
        try:
            # 保存为JSON文件
            with open(CANDIDATE_EXTRACTION_NER_LATEST_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存为带时间戳的归档文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_file = os.path.join(CANDIDATE_EXTRACTION_DIR, f"candidate_entity_extraction_{timestamp}.json")
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存为Excel文件
            self._save_excel(results, CANDIDATE_EXTRACTION_NER_LATEST_XLSX)
            
            logger.info(f"已保存 {len(results)} 条记录到 {CANDIDATE_EXTRACTION_NER_LATEST_JSON} 和 {archive_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
        
        self.performance['save_time'] += time.time() - start_time
    
    def _save_excel(self, results, output_file):
        """保存为Excel文件
        
        Args:
            results (dict): 处理结果字典
            output_file (str): 输出文件路径
        """
        rows = []
        for pmid, data in results.items():
            for topic, ner_result in data.get('ner_results', {}).items():
                for entity in ner_result.get('entities', []):
                    rows.append({
                        "pmid": pmid,
                        "abstract": data['abstract'],
                        "topic": topic,
                        "entity": entity['entity'],
                        "type": entity['type'],
                        "confidence": entity['confidence'],
                        "votes": entity['votes']
                    })
        
        df = pd.DataFrame(rows)
        df.to_excel(output_file, index=False)


def run_ner_with_reflection(pmid: str, abstract: str, seed_entities: Dict, 
                            topic: str, reflection_report: Optional[str] = None) -> Dict:
    """可被外部调用的NER接口（支持反思报告）
    
    Args:
        pmid: 文献ID
        abstract: 摘要文本
        seed_entities: 种子实体 {"Food": [], "Disease": [], "Nutrient": [], "DietaryPattern": []}
        topic: 主题（如"T1"）
        reflection_report: 反思报告（可选）
        
    Returns:
        Dict: {
            "entities": [...],  # 实体列表
            "raw_votes": {...},  # 原始投票结果
            "p_eh": float,  # 头实体置信度（如果有）
            "p_et": float   # 尾实体置信度（如果有）
        }
    """
    agent = NERAgent()
    
    data = {
        "abstract": abstract,
        "seed_entities": seed_entities,
        "reflection_report": reflection_report or ""
    }
    
    result = agent.process_single_record_topic(pmid, data, topic)
    
    if reflection_report:
        result["reflection_report"] = reflection_report
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='NER Agent: 实体识别智能体')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量（用于批量运行控制）')
    parser.add_argument('--parallel', action='store_true', default=True, help='是否并行处理')
    args = parser.parse_args()
    
    agent = NERAgent()
    agent.run(use_parallel=args.parallel, max_workers=MAX_WORKERS, batch_size=BATCH_SIZE, max_records=args.limit)


