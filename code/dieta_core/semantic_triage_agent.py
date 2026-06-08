#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_triage.py - Triage Agent：分诊智能体，主题感知筛选

功能：
- 读取输入的PubMed摘要数据
- 使用LLM进行主题感知，判断每条摘要属于哪些主题簇
- 输出带时间戳的归档文件和固定名latest文件
- 支持断点续跑

按照DIETA框架实现 - Triage Agent（分诊智能体）

输入：
- data/diabetes_diet_papers.csv（列：pmid, title, abstract, pubdate, query_source）

输出：
- results/semantic_triage/semantic_triage_latest.xlsx（固定名，每次覆盖）
- results/semantic_triage/semantic_triage_[YYYYMMDD_HHMMSS].xlsx（带时间戳归档）
"""

import os
import time
import signal
import pandas as pd
import requests
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_client import LLMClient
from settings import (
    API_KEY, API_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    API_RETRY_TIMES, API_RETRY_INTERVAL, INPUT_CSV, ROUTER_PROMPT,
    SEMANTIC_TRIAGE_DIR, SEMANTIC_TRIAGE_LATEST, MAX_WORKERS
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SEMANTIC_TRIAGE_DIR, 'semantic_triage.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保结果目录存在
os.makedirs(SEMANTIC_TRIAGE_DIR, exist_ok=True)

class TriageAgent:
    """Triage Agent：分诊智能体，主题感知筛选"""
    
    def __init__(self):
        """初始化Triage Agent"""
        # 读取提示词
        with open(ROUTER_PROMPT, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read().strip()
        # 初始化状态
        self.llm_client = LLMClient()
        self.running = True
        self.results = []
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.info(f"收到信号 {signum}，准备保存进度并退出")
        self.running = False
    
    def load_input_data(self):
        """加载输入数据
        
        Returns:
            pd.DataFrame: 输入数据
        """
        try:
            # 尝试不同的编码方式
            encodings = ['utf-8-sig', 'gbk', 'latin1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(INPUT_CSV, encoding=encoding)
                    logger.info(f"成功加载输入数据，共 {len(df)} 条记录，使用编码: {encoding}")
                    return df
                except Exception as e:
                    logger.warning(f"使用 {encoding} 编码失败: {e}")
            
            # 如果所有编码都失败，使用二进制模式读取并处理
            logger.info("所有编码尝试失败，使用二进制模式读取")
            with open(INPUT_CSV, 'rb') as f:
                content = f.read()
            # 尝试解码，替换错误
            content_str = content.decode('utf-8', errors='replace')
            # 写入中间缓存文件
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_file.write(content_str)
                temp_path = temp_file.name
            # 读取中间缓存文件
            df = pd.read_csv(temp_path)
            os.unlink(temp_path)
            logger.info(f"成功加载输入数据，共 {len(df)} 条记录，使用二进制处理")
            return df
        except Exception as e:
            logger.error(f"加载输入数据失败: {e}")
            raise
    
    def load_processed_pmids(self):
        """加载已处理的pmid
        
        Returns:
            set: 已处理的pmid集合
        """
        processed_pmids = set()
        if os.path.exists(SEMANTIC_TRIAGE_LATEST):
            try:
                df = pd.read_excel(SEMANTIC_TRIAGE_LATEST)
                processed_pmids = set(df['pmid'].astype(str).tolist())
                logger.info(f"已处理 {len(processed_pmids)} 条记录")
            except Exception as e:
                logger.warning(f"读取已处理记录失败: {e}")
        return processed_pmids
    
    def build_prompt(self, abstract):
        """构建提示词
        
        Args:
            abstract (str): 摘要原文
            
        Returns:
            str: 完整的提示词
        """
        # 限制摘要长度，避免API调用失败
        MAX_ABSTRACT_LENGTH = 20000
        if len(abstract) > MAX_ABSTRACT_LENGTH:
            abstract = abstract[:MAX_ABSTRACT_LENGTH] + "..."
            logger.warning(f"摘要长度超过 {MAX_ABSTRACT_LENGTH} 字符，已截断")
        
        return f"Abstract: {abstract}\n\n{self.prompt_template}"
    
    def call_llm(self, prompt):
        """Call the LLM through the centralized DIETA LLM gateway.

        Keeping this method name preserves the original output format and call sites,
        while the implementation now follows an agent-style Tool/Gateway pattern.
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
    
    def parse_llm_response(self, response):
        """解析LLM返回结果
        
        Args:
            response (str): LLM返回结果
            
        Returns:
            tuple: (t1, t2, t3, t4, t5, t6, matched_clusters, parse_error)
        """
        if not response:
            return (0, 0, 0, 0, 0, 0, "", True)
        
        # 提取六位二进制数字（支持带逗号和不带逗号两种格式）
        import re
        # 尝试匹配带逗号的格式，如 "0,0,0,0,0,0"
        match = re.search(r'([01]),([01]),([01]),([01]),([01]),([01])', response)
        if match:
            # 解析带逗号的格式
            t1 = int(match.group(1))
            t2 = int(match.group(2))
            t3 = int(match.group(3))
            t4 = int(match.group(4))
            t5 = int(match.group(5))
            t6 = int(match.group(6))
        else:
            # 尝试匹配不带逗号的格式，如 "000000"
            match = re.search(r'([01]{6})', response)
            if not match:
                return (0, 0, 0, 0, 0, 0, "", True)
            
            binary_str = match.group(1)
            if len(binary_str) != 6:
                return (0, 0, 0, 0, 0, 0, "", True)
            
            # 解析二进制字符串
            t1 = int(binary_str[0])
            t2 = int(binary_str[1])
            t3 = int(binary_str[2])
            t4 = int(binary_str[3])
            t5 = int(binary_str[4])
            t6 = int(binary_str[5])
        
        # 生成matched_clusters
        matched_clusters = []
        if t1:
            matched_clusters.append("T1")
        if t2:
            matched_clusters.append("T2")
        if t3:
            matched_clusters.append("T3")
        if t4:
            matched_clusters.append("T4")
        if t5:
            matched_clusters.append("T5")
        if t6:
            matched_clusters.append("T6")
        
        matched_clusters_str = ",".join(matched_clusters)
        
        return (t1, t2, t3, t4, t5, t6, matched_clusters_str, False)
    
    def process_single_record(self, pmid, abstract):
        """处理单条记录
        
        Args:
            pmid (str): 文献ID
            abstract (str): 摘要原文
            
        Returns:
            dict: 处理结果
        """
        try:
            # Ablation: remove ontology-driven routing by activating all task routes.
            # This preserves the legacy output table while changing only route activation.
            if os.environ.get("GCDI_ABLATION_DISABLE_ROUTING", "0").strip().lower() in {"1", "true", "yes", "on"}:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                return {
                    "pmid": pmid,
                    "abstract": abstract,
                    "T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 1, "T6": 1,
                    "matched_clusters": "T1,T2,T3,T4,T5,T6",
                    "parse_error": False,
                    "timestamp": timestamp
                }

            # 构建提示词
            prompt = self.build_prompt(abstract)
            
            # 调用LLM
            response = self.call_llm(prompt)
            
            # 记录LLM原始输出
            logger.info(f"PMID {pmid} semantic triage completed")
            
            # 解析结果
            t1, t2, t3, t4, t5, t6, matched_clusters, parse_error = self.parse_llm_response(response)
            
            # 生成时间戳
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "pmid": pmid,
                "abstract": abstract,
                "T1": t1,
                "T2": t2,
                "T3": t3,
                "T4": t4,
                "T5": t5,
                "T6": t6,
                "matched_clusters": matched_clusters,
                "parse_error": parse_error,
                "timestamp": timestamp
            }
        except Exception as e:
            logger.error(f"处理记录 {pmid} 失败: {e}")
            # 出错时返回默认值
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            return {
                "pmid": pmid,
                "abstract": abstract,
                "T1": 0,
                "T2": 0,
                "T3": 0,
                "T4": 0,
                "T5": 0,
                "T6": 0,
                "matched_clusters": "",
                "parse_error": True,
                "timestamp": timestamp
            }
    
    def run(self, max_records=None):
        """运行Triage Agent
        
        Args:
            max_records (int, optional): 最大处理记录数. Defaults to None.
        """
        logger.info("开始运行Triage Agent（分诊智能体）")
        
        # 加载输入数据
        df_input = self.load_input_data()
        
        # 加载已处理的pmid
        processed_pmids = self.load_processed_pmids()
        
        # 获取pmid列的索引（第一列）
        pmid_col = df_input.columns[0]
        abstract_col = 'abstract' if 'abstract' in df_input.columns else df_input.columns[2]
        
        # 过滤出未处理的记录
        df_to_process = df_input[~df_input[pmid_col].astype(str).isin(processed_pmids)]
        
        # 限制处理记录数量
        if max_records:
            df_to_process = df_to_process.head(max_records)
            logger.info(f"限制处理前 {max_records} 条记录")
        
        logger.info(f"待处理 {len(df_to_process)} 条记录")
        
        # 处理记录
        self.results = []
        
        # 如果有已处理的记录，先加载它们
        if os.path.exists(SEMANTIC_TRIAGE_LATEST):
            try:
                df_existing = pd.read_excel(SEMANTIC_TRIAGE_LATEST)
                self.results = df_existing.to_dict('records')
                logger.info(f"加载了 {len(self.results)} 条已处理记录")
            except Exception as e:
                logger.warning(f"加载已处理记录失败: {e}")
        
        # 处理未处理的记录
        processed_count = 0
        failed_records = []
        
        # 准备待处理的记录列表
        records_to_process = []
        for _, row in df_to_process.iterrows():
            pmid = str(row[pmid_col])
            abstract = str(row[abstract_col])
            records_to_process.append((pmid, abstract))
        
        # 第一次处理 - 并发
        if records_to_process and self.running:
            logger.info(f"开始并发处理 {len(records_to_process)} 条记录，使用 {MAX_WORKERS} 个线程")
            
            # 批量提交任务，避免一次性提交所有任务导致内存问题
            BATCH_SUBMIT_SIZE = 50
            total_records = len(records_to_process)
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_record = {}
                processed_count = 0
                
                # 分批提交任务
                for i in range(0, total_records, BATCH_SUBMIT_SIZE):
                    if not self.running:
                        break
                    
                    batch_records = records_to_process[i:i+BATCH_SUBMIT_SIZE]
                    logger.info(f"提交第 {i//BATCH_SUBMIT_SIZE + 1} 批任务，共 {len(batch_records)} 条记录")
                    
                    # 提交当前批次的任务
                    for pmid, abstract in batch_records:
                        future = executor.submit(self.process_single_record, pmid, abstract)
                        future_to_record[future] = (pmid, abstract)
                
                # 处理完成的任务
                for future in tqdm(as_completed(future_to_record), total=len(future_to_record), desc="处理摘要"):
                    # 检查是否收到中止信号
                    if not self.running:
                        logger.info("收到中止信号，取消所有任务")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    try:
                        result = future.result()
                        self.results.append(result)
                        processed_count += 1
                        
                        # 检查是否处理失败
                        if result['parse_error'] and result['matched_clusters'] == "":
                            pmid, abstract = future_to_record[future]
                            failed_records.append((pmid, abstract))
                            logger.warning(f"记录 {pmid} 处理失败，将在第一轮处理后重试")
                        
                        # 每处理10条记录就保存一次，确保进度不会丢失
                        if processed_count % 10 == 0:
                            self.save_results(self.results)
                            logger.info(f"已处理 {processed_count} 条记录，共 {total_records} 条")
                    except Exception as e:
                        logger.error(f"处理任务失败: {e}")
                        # 尝试获取失败的记录信息
                        try:
                            pmid, abstract = future_to_record.get(future, ("未知", ""))
                            logger.error(f"失败的记录: PMID={pmid}")
                            # 将失败的记录添加到重试列表
                            failed_records.append((pmid, abstract))
                        except Exception:
                            pass
            
            # 保存最终结果
            if self.results:
                self.save_results(self.results)
        
        # 重试失败的记录 - 并发
        if failed_records and self.running:
            logger.info(f"开始并发重试 {len(failed_records)} 条失败的记录")
            retry_count = 0
            total_failed = len(failed_records)
            
            # 批量提交任务
            BATCH_SUBMIT_SIZE = 50
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_record = {}
                
                # 分批提交任务
                for i in range(0, total_failed, BATCH_SUBMIT_SIZE):
                    if not self.running:
                        break
                    
                    batch_records = failed_records[i:i+BATCH_SUBMIT_SIZE]
                    logger.info(f"提交第 {i//BATCH_SUBMIT_SIZE + 1} 批重试任务，共 {len(batch_records)} 条记录")
                    
                    # 提交当前批次的任务
                    for pmid, abstract in batch_records:
                        future = executor.submit(self.process_single_record, pmid, abstract)
                        future_to_record[future] = (pmid, abstract)
                
                # 处理完成的任务
                for future in tqdm(as_completed(future_to_record), total=len(future_to_record), desc="重试失败记录"):
                    # 检查是否收到中止信号
                    if not self.running:
                        logger.info("收到中止信号，取消所有任务")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    try:
                        result = future.result()
                        pmid, _ = future_to_record[future]
                        
                        # 找到原记录的索引并更新
                        updated = False
                        for i, existing_result in enumerate(self.results):
                            if existing_result['pmid'] == pmid:
                                self.results[i] = result
                                updated = True
                                break
                        
                        if not updated:
                            logger.warning(f"未找到记录 {pmid} 的原始结果，将新结果添加到列表")
                            self.results.append(result)
                        
                        retry_count += 1
                        
                        # 每重试5条记录就保存一次
                        if retry_count % 5 == 0:
                            self.save_results(self.results)
                            logger.info(f"已重试 {retry_count} 条记录，共 {total_failed} 条")
                    except Exception as e:
                        logger.error(f"重试任务失败: {e}")
                        # 尝试获取失败的记录信息
                        try:
                            pmid, abstract = future_to_record.get(future, ("未知", ""))
                            logger.error(f"重试失败的记录: PMID={pmid}")
                        except Exception:
                            pass
            
            # 保存最终结果
            if self.results:
                self.save_results(self.results)
        
        # 保存最终结果
        if self.results:
            self.save_results(self.results)
        
        if not self.running:
            logger.info("Triage Agent被中止")
        else:
            logger.info("Triage Agent运行完成")
    
    def save_results(self, results):
        """保存结果
        
        Args:
            results (list): 处理结果列表
        """
        try:
            # 转换为DataFrame
            df_results = pd.DataFrame(results)
            
            # 保存为latest文件
            df_results.to_excel(SEMANTIC_TRIAGE_LATEST, index=False)
            
            # 保存为带时间戳的归档文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_file = os.path.join(SEMANTIC_TRIAGE_DIR, f"semantic_triage_{timestamp}.xlsx")
            df_results.to_excel(archive_file, index=False)
            
            logger.info(f"已保存 {len(results)} 条记录到 {SEMANTIC_TRIAGE_LATEST} 和 {archive_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Triage Agent: 主题分诊智能体')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量（用于批量运行控制）')
    args = parser.parse_args()
    
    agent = TriageAgent()
    agent.run(max_records=args.limit)



