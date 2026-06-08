#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
candidate_relation_extraction.py - Extraction Agent（RE部分）：LLM多提示词关系抽取

功能：
- 读取candidate_entity_extraction的输出结果
- 使用多个提示词进行关系抽取
- 实现自一致性投票
- 输出带时间戳的归档文件和固定名latest文件
- 支持断点续跑

输入：
- results/candidate_extraction/candidate_entity_extraction_latest.json

输出：
- results/candidate_extraction/candidate_relation_extraction_latest.json（固定名，每次覆盖）
- results/candidate_extraction/candidate_relation_extraction_[时间戳].json（带时间戳归档）
- results/candidate_extraction/candidate_relation_extraction_latest.xlsx（Excel展平版，供人工查看）
"""

import os
import time
import signal
import json
import pandas as pd
import requests
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_client import LLMClient
from settings import (
    API_KEY, API_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    API_RETRY_TIMES, API_RETRY_INTERVAL, BATCH_SIZE, RE_PROMPT_COUNT, DELTA,
    CANDIDATE_EXTRACTION_NER_LATEST_JSON, CANDIDATE_EXTRACTION_DIR, CANDIDATE_EXTRACTION_RE_LATEST_JSON, CANDIDATE_EXTRACTION_RE_LATEST_XLSX,
    TOPIC_CONFIG, PROMPTS_DIR, USE_PARALLEL, MAX_WORKERS
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CANDIDATE_EXTRACTION_DIR, 'candidate_relation_extraction.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保结果目录存在
os.makedirs(CANDIDATE_EXTRACTION_DIR, exist_ok=True)

class REAgent:
    """RE Agent：LLM多提示词关系抽取"""
    
    def __init__(self):
        """初始化RE Agent"""
        # 加载提示词
        self.llm_client = LLMClient()
        self.prompts = self._load_prompts()
        # 初始化状态
        self.running = True
        self.results = {}
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.info(f"收到信号 {signum}，准备保存进度并退出")
        self.running = False
    
    def _load_prompts(self):
        """加载提示词
        
        Returns:
            dict: 提示词字典
        """
        prompts = {}
        for topic in TOPIC_CONFIG.keys():
            prompts[topic] = {}
            for i in range(1, RE_PROMPT_COUNT + 1):
                prompt_file = os.path.join(PROMPTS_DIR, f"re_{topic}_q{i}.txt")
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        prompts[topic][f"q{i}"] = f.read().strip()
                else:
                    logger.warning(f"提示词文件不存在: {prompt_file}")
        return prompts
    
    def load_input_data(self):
        """加载输入数据
        
        Returns:
            dict: 输入数据
        """
        try:
            with open(CANDIDATE_EXTRACTION_NER_LATEST_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
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
        if os.path.exists(CANDIDATE_EXTRACTION_RE_LATEST_JSON):
            try:
                with open(CANDIDATE_EXTRACTION_RE_LATEST_JSON, 'r', encoding='utf-8') as f:
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
        # 构建Validation Records部分
        batch_str = "Validation Records:\n[\n"
        for item in batch_data:
            batch_str += "  {\n"
            batch_str += f"    \"row_id\": \"{item['row_id']}\",\n"
            batch_str += f"    \"abstract\": \"{item['abstract']}\",\n"
            batch_str += "    \"entities\": [\n"
            for entity in item['entities']:
                batch_str += f"      {{\"entity\": \"{entity['entity']}\", \"type\": \"{entity['type']}\"}},\n"
            # 移除最后一个逗号
            if item['entities']:
                batch_str = batch_str.rstrip(",\n") + "\n"
            batch_str += "    ]\n"
            batch_str += "  },\n"
        # 移除最后一个逗号
        if batch_data:
            batch_str = batch_str.rstrip(",\n") + "\n"
        batch_str += "]\n\n"
        
        # Feedback channel: optional reflection reports are injected only
        # when the Feedback Controller explicitly requests relation re-evaluation.
        reflection_blocks = []
        for item in batch_data:
            if item.get('reflection_report'):
                reflection_blocks.append(
                    f"PMID/row_id: {item.get('row_id', '')}\n"
                    f"{item.get('reflection_report', '')}"
                )
        if reflection_blocks:
            batch_str += "\n[Feedback / Reflection Guidance]\n"
            batch_str += "\n\n".join(reflection_blocks)[:4000]
            batch_str += "\nPlease re-evaluate relation extraction under the same legal relation schema and output JSON format.\n\n"

        # 添加固定指令部分
        prompt_content = self.prompts.get(topic, {}).get(prompt_key, "")
        batch_str += prompt_content
        
        return batch_str
    
    def call_llm(self, prompt):
        """Call the LLM through the centralized DIETA LLM gateway.

        Method name and return contract are preserved, while model access is now
        managed as a registered agent tool/gateway rather than raw scattered API calls.
        """
        try:
            return self.llm_client.generate(
                prompt=prompt,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                use_cache=True,
            ).strip()
        except Exception as e:
            logger.error(f"LLM gateway call failed: {e}")
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
            dict: 解析后的关系结果
        """
        results = {}
        if not response:
            # 如果返回结果为空，所有关系为空列表
            for item in batch_data:
                results[item['row_id']] = []
            return results
        
        # 尝试解析JSON格式的返回结果
        try:
            # 提取JSON部分
            import re
            
            # 1. 处理Markdown代码块（Qwen2.5-72B格式）
            code_block_match = re.search(r'```json\n([\s\S]*?)\n```', response)
            if code_block_match:
                response = code_block_match.group(1)
                logger.info("找到并处理Markdown代码块")
            
            # 2. 寻找完整的JSON数组
            json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    triples_list = json.loads(json_str)
                    
                    # 按row_id组织结果
                    for item in triples_list:
                        row_id = item.get('row_id')
                        triples = item.get('triples', [])
                        results[row_id] = triples
                except Exception as e:
                    logger.warning(f"解析JSON数组失败，尝试修复: {e}")
                    # 尝试修复JSON
                    repaired_json = self.repair_json(json_str)
                    try:
                        triples_list = json.loads(repaired_json)
                        
                        # 按row_id组织结果
                        for item in triples_list:
                            row_id = item.get('row_id')
                            triples = item.get('triples', [])
                            results[row_id] = triples
                        logger.info("JSON修复成功")
                    except Exception as e2:
                        logger.error(f"修复JSON后仍然失败: {e2}")
                        logger.error(f"原始JSON: {json_str[:300]}...")
                        logger.error(f"修复后JSON: {repaired_json[:300]}...")
            
            # 确保所有batch_data中的row_id都有结果
            for item in batch_data:
                if item['row_id'] not in results:
                    results[item['row_id']] = []
        except Exception as e:
            logger.error(f"解析LLM返回结果失败: {e}")
            # 解析失败时，所有关系为空列表
            for item in batch_data:
                results[item['row_id']] = []
        
        return results
    
    def calculate_votes(self, all_triples, entities=None):
        """计算关系投票
        
        Args:
            all_triples (dict): 所有提示词的关系结果
            entities (list, optional): 实体列表，包含实体的置信度信息
            
        Returns:
            list: 关系投票结果
        """
        # 统计每个三元组的出现次数
        triple_counts = {}
        for prompt_key, triples in all_triples.items():
            for triple in triples:
                # 构建三元组的唯一标识
                triple_key = (triple['subject'], triple['relation'], triple['object'])
                if triple_key not in triple_counts:
                    triple_counts[triple_key] = {
                        'subject': triple['subject'],
                        'relation': triple['relation'],
                        'object': triple['object'],
                        'count': 0,
                        'prompts': []
                    }
                triple_counts[triple_key]['count'] += 1
                triple_counts[triple_key]['prompts'].append(prompt_key)
        
        # 计算置信度并过滤
        final_triples = []
        for triple_key, info in triple_counts.items():
            confidence = info['count'] / RE_PROMPT_COUNT
            if confidence >= DELTA:
                triple_result = {
                    "subject": info['subject'],
                    "relation": info['relation'],
                    "object": info['object'],
                    "confidence": confidence,
                    "votes": info['count'],
                    "raw_votes": {},
                    "subject_confidence": 0.8,  # 实体置信度，默认0.8
                    "object_confidence": 0.8,   # 实体置信度，默认0.8
                    "subject_type": None,         # 实体类型，默认None
                    "object_type": None,          # 实体类型，默认None
                    "abstract": None             # 文本关联
                }
                
                # 添加原始投票结果
                for prompt_key, triples in all_triples.items():
                    for triple in triples:
                        if (triple['subject'], triple['relation'], triple['object']) == triple_key:
                            triple_result['raw_votes'][prompt_key] = triple['relation']
                
                # 添加实体置信度和类型信息，支持容错匹配
                if entities:
                    for entity in entities:
                        # 标准化实体名称：忽略大小写，连字符和空格等价
                        def normalize(s):
                            return s.lower().replace('-', ' ').replace('  ', ' ').strip()
                        
                        if normalize(entity['entity']) == normalize(info['subject']):
                            triple_result['subject_confidence'] = entity['confidence']
                            triple_result['subject_type'] = entity['type']
                        if normalize(entity['entity']) == normalize(info['object']):
                            triple_result['object_confidence'] = entity['confidence']
                            triple_result['object_type'] = entity['type']
                
                final_triples.append(triple_result)
        
        return final_triples
    
    def process_batch(self, topic, batch_data):
        """处理批数据
        
        Args:
            topic (str): 主题
            batch_data (list): 批处理数据
            
        Returns:
            dict: 处理结果
        """
        results = {}
        
        # 对每个提示词运行LLM
        all_triples = {}
        for prompt_key in self.prompts.get(topic, {}).keys():
            # 构建提示词
            prompt = self.build_batch_prompt(topic, batch_data, prompt_key)
            
            # 调用LLM
            response = self.call_llm(prompt)
            
            # 解析结果
            prompt_results = self.parse_llm_response(response, batch_data)
            all_triples[prompt_key] = prompt_results
        
        # 对每个摘要计算投票结果
        for item in batch_data:
            row_id = item['row_id']
            
            # 收集该摘要的所有三元组
            abstract_triples = {}
            for prompt_key, prompt_results in all_triples.items():
                abstract_triples[prompt_key] = prompt_results.get(row_id, [])
            
            # 计算投票结果，传递实体信息
            final_triples = self.calculate_votes(abstract_triples, item.get('entities', []))
            
            # 添加文本关联信息
            for triple in final_triples:
                triple['abstract'] = item.get('abstract', '')
            
            results[row_id] = final_triples
        
        return results
    
    def process_single_record(self, pmid, data):
        """处理单条记录
        
        Args:
            pmid (str): 文献ID
            data (dict): 记录数据
            
        Returns:
            dict: 处理结果
        """
        try:
            re_results = {}
            
            # 对每个命中的主题进行处理
            for topic in data['matched_clusters']:
                if topic in self.prompts:
                    # 获取该主题的实体
                    ner_result = data['ner_results'].get(topic, {})
                    entities = ner_result.get('entities', [])
                    
                    if entities:
                        # 构建批数据
                        batch_data = [{
                            "row_id": pmid,
                            "abstract": data['abstract'],
                            "entities": entities
                        }]
                        
                        # 处理批数据
                        batch_results = self.process_batch(topic, batch_data)
                        
                        if pmid in batch_results:
                            re_results[topic] = {
                                "triples": batch_results[pmid],
                                "entities": entities  # 保留实体信息
                            }
            
            result = {
                "pmid": pmid,
                "abstract": data['abstract'],  # 保留摘要信息
                "re_results": re_results,
                "processed": True,
                "error": None
            }
            
            return result
        except Exception as e:
            logger.error(f"处理记录 {pmid} 失败: {e}")
            return {
                "pmid": pmid,
                "abstract": data.get('abstract', ''),  # 保留摘要信息
                "re_results": {},
                "processed": False,
                "error": str(e)
            }
    
    def run(self, use_parallel=True, max_workers=5, save_interval=10, max_records=None):
        """运行RE Agent
        
        Args:
            use_parallel (bool): 是否使用并行处理
            max_workers (int): 最大并发数
            save_interval (int): 保存间隔（每处理多少条记录保存一次）
            max_records (int, optional): 最大处理记录数. Defaults to None.
        """
        logger.info("开始运行RE Agent")
        
        # 加载输入数据
        input_data = self.load_input_data()
        
        # 加载已处理的pmid
        processed_pmids = self.load_processed_pmids()
        
        # 过滤出未处理的记录
        to_process = {pmid: data for pmid, data in input_data.items() if pmid not in processed_pmids}
        
        # 限制处理记录数量
        if max_records:
            to_process = dict(list(to_process.items())[:max_records])
            logger.info(f"限制处理前 {max_records} 条记录")
        
        logger.info(f"待处理 {len(to_process)} 条记录")
        
        # 处理记录
        self.results = {}
        
        # 如果有已处理的记录，先加载它们
        if os.path.exists(CANDIDATE_EXTRACTION_RE_LATEST_JSON):
            try:
                with open(CANDIDATE_EXTRACTION_RE_LATEST_JSON, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                logger.info(f"加载了 {len(self.results)} 条已处理记录")
            except Exception as e:
                logger.warning(f"加载已处理记录失败: {e}")
        
        # 处理未处理的记录
        processed_count = 0
        to_process_list = list(to_process.items())
        
        if use_parallel and len(to_process_list) > 0:
            logger.info(f"使用并行处理，最大并发数: {max_workers}")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_pmid = {
                    executor.submit(self.process_single_record, pmid, data): pmid
                    for pmid, data in to_process_list
                }
                
                # 收集结果
                for future in tqdm(as_completed(future_to_pmid), total=len(future_to_pmid), desc="抽取关系"):
                    # 检查是否收到中止信号
                    if not self.running:
                        logger.info("收到中止信号，停止处理")
                        break
                    
                    try:
                        pmid = future_to_pmid[future]
                        result = future.result()
                        self.results[pmid] = result
                        processed_count += 1
                        
                        # 每处理save_interval条记录保存一次
                        if processed_count % save_interval == 0:
                            self.save_results(self.results)
                            logger.info(f"已处理 {processed_count} 条记录，共 {len(to_process)} 条")
                    except Exception as e:
                        logger.error(f"处理记录时出错: {e}")
        else:
            logger.info("使用串行处理")
            for pmid, data in tqdm(to_process_list, total=len(to_process_list), desc="抽取关系"):
                # 检查是否收到中止信号
                if not self.running:
                    logger.info("收到中止信号，停止处理")
                    break
                
                result = self.process_single_record(pmid, data)
                self.results[pmid] = result
                processed_count += 1
                
                # 每处理save_interval条记录保存一次
                if processed_count % save_interval == 0:
                    self.save_results(self.results)
                    logger.info(f"已处理 {processed_count} 条记录，共 {len(to_process)} 条")
        
        # 保存最终结果
        if self.results:
            self.save_results(self.results)
        
        if not self.running:
            logger.info("RE Agent被中止")
        else:
            logger.info("RE Agent运行完成")
    
    def save_results(self, results):
        """保存结果
        
        Args:
            results (dict): 处理结果字典
        """
        try:
            # 保存为JSON文件
            with open(CANDIDATE_EXTRACTION_RE_LATEST_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存为带时间戳的归档文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_file = os.path.join(CANDIDATE_EXTRACTION_DIR, f"candidate_relation_extraction_{timestamp}.json")
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存为Excel文件
            self._save_excel(results, CANDIDATE_EXTRACTION_RE_LATEST_XLSX)
            
            logger.info(f"已保存 {len(results)} 条记录到 {CANDIDATE_EXTRACTION_RE_LATEST_JSON} 和 {archive_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _save_excel(self, results, output_file):
        """保存为Excel文件
        
        Args:
            results (dict): 处理结果字典
            output_file (str): 输出文件路径
        """
        rows = []
        for pmid, data in results.items():
            for topic, re_result in data.get('re_results', {}).items():
                for triple in re_result.get('triples', []):
                    rows.append({
                        "pmid": pmid,
                        "topic": topic,
                        "subject": triple['subject'],
                        "relation": triple['relation'],
                        "object": triple['object'],
                        "confidence": triple['confidence'],
                        "votes": triple['votes'],
                        "subject_confidence": triple.get('subject_confidence'),
                        "object_confidence": triple.get('object_confidence'),
                        "abstract": triple.get('abstract', '')[:100]  # 保存摘要的前100个字符
                    })
        
        df = pd.DataFrame(rows)
        df.to_excel(output_file, index=False)


def run_re_with_reflection(pmid: str, abstract: str, entities: List[Dict], 
                           topic: str, reflection_report: Optional[str] = None) -> Dict:
    """可被外部调用的RE接口（支持反思报告）
    
    Args:
        pmid: 文献ID
        abstract: 摘要文本
        entities: 实体列表 [{"entity": str, "type": str, "confidence": float}, ...]
        topic: 主题（如"T1"）
        reflection_report: 反思报告（可选）
        
    Returns:
        Dict: {
            "triples": [...],  # 三元组列表
            "raw_votes": {...},  # 原始投票结果
            "p_r": float,  # 关系置信度
            "U_t": float   # 不确定性分数（如果有）
        }
    """
    agent = REAgent()
    
    batch_data = [{
        "row_id": pmid,
        "abstract": abstract,
        "entities": entities
    }]
    
    if reflection_report:
        batch_data[0]["reflection_report"] = reflection_report
    
    results = agent.process_batch(topic, batch_data)
    
    return {
        "triples": results.get(pmid, []),
        "pmid": pmid,
        "topic": topic
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='RE Agent: 关系抽取智能体')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量（用于批量运行控制）')
    args = parser.parse_args()
    
    agent = REAgent()
    agent.run(use_parallel=USE_PARALLEL, max_workers=MAX_WORKERS, max_records=args.limit)


