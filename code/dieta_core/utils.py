#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - 公共工具模块

提供项目中重复使用的功能：
- 日志配置
- JSON修复
- 断点续跑
- 结果保存
- 实体标准化
"""

import os
import time
import json
import re
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Set


def setup_logging(log_dir: str, log_filename: str, level: int = logging.INFO) -> logging.Logger:
    """配置日志
    
    Args:
        log_dir: 日志目录
        log_filename: 日志文件名
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger(log_filename.replace('.log', ''))
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def repair_json(json_str: str) -> str:
    """尝试修复损坏的JSON字符串
    
    Args:
        json_str: 损坏的JSON字符串
        
    Returns:
        str: 修复后的JSON字符串
    """
    repaired = json_str
    
    open_brackets = repaired.count('[')
    close_brackets = repaired.count(']')
    open_braces = repaired.count('{')
    close_braces = repaired.count('}')
    
    while close_brackets < open_brackets:
        repaired = repaired.rstrip() + ']'
        close_brackets += 1
    
    while close_braces < open_braces:
        repaired = repaired.rstrip() + '}'
        close_braces += 1
    
    repaired = re.sub(r',\s*(\]|\})', r'\1', repaired)
    repaired = re.sub(r',\s*(\})', r'\1', repaired)
    repaired = re.sub(r'"\.\.\."', '""', repaired)
    repaired = re.sub(r'\.\.\."', '"', repaired)
    repaired = re.sub(r'"\.\.\.', '"', repaired)
    
    quote_count = repaired.count('"')
    if quote_count % 2 != 0:
        last_quote_pos = repaired.rfind('"')
        if last_quote_pos != -1:
            repaired = repaired[:last_quote_pos + 1]
    
    return repaired


def extract_json_from_text(text: str) -> str:
    """从文本中提取JSON字符串
    
    Args:
        text: 包含JSON的文本
        
    Returns:
        str: 提取的JSON字符串
    """
    text = text.strip()
    
    code_block_match = re.search(r'```json\n([\s\S]*?)\n```', text)
    if code_block_match:
        return code_block_match.group(1)
    
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace+1]
    
    return text


def load_processed_ids(latest_file: str, id_key: str = 'pmid') -> Set[str]:
    """加载已处理的ID集合
    
    Args:
        latest_file: latest文件路径
        id_key: ID字段名
        
    Returns:
        Set[str]: 已处理的ID集合
    """
    processed_ids = set()
    if os.path.exists(latest_file):
        try:
            if latest_file.endswith('.json'):
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                processed_ids = set(data.keys())
            elif latest_file.endswith('.xlsx'):
                df = pd.read_excel(latest_file)
                if id_key in df.columns:
                    processed_ids = set(df[id_key].astype(str).tolist())
        except Exception as e:
            logging.warning(f"读取已处理记录失败: {e}")
    return processed_ids


def load_existing_results(latest_file: str) -> Any:
    """加载已有的处理结果
    
    Args:
        latest_file: latest文件路径
        
    Returns:
        Any: 已有的结果数据
    """
    if not os.path.exists(latest_file):
        return None
    
    try:
        if latest_file.endswith('.json'):
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif latest_file.endswith('.xlsx'):
            return pd.read_excel(latest_file)
    except Exception as e:
        logging.warning(f"加载已处理记录失败: {e}")
    return None


def save_results_with_timestamp(data: Any, 
                                 latest_file: str, 
                                 archive_dir: str, 
                                 file_prefix: str,
                                 save_excel: bool = False,
                                 excel_converter=None) -> None:
    """保存结果（同时保存latest和带时间戳的归档）
    
    Args:
        data: 要保存的数据
        latest_file: latest文件路径
        archive_dir: 归档目录
        file_prefix: 文件名前缀
        save_excel: 是否同时保存Excel
        excel_converter: 数据转换为Excel行的函数
    """
    os.makedirs(archive_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if latest_file.endswith('.json'):
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        archive_file = os.path.join(archive_dir, f"{file_prefix}_{timestamp}.json")
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif latest_file.endswith('.xlsx'):
        if isinstance(data, pd.DataFrame):
            data.to_excel(latest_file, index=False)
            archive_file = os.path.join(archive_dir, f"{file_prefix}_{timestamp}.xlsx")
            data.to_excel(archive_file, index=False)
    
    if save_excel and excel_converter and latest_file.endswith('.json'):
        excel_file = latest_file.replace('.json', '.xlsx')
        rows = excel_converter(data)
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(excel_file, index=False)
            archive_excel = os.path.join(archive_dir, f"{file_prefix}_{timestamp}.xlsx")
            df.to_excel(archive_excel, index=False)


def normalize_entity(entity: str) -> str:
    """标准化实体
    
    Args:
        entity: 实体字符串
        
    Returns:
        str: 标准化后的实体
    """
    entity = re.sub(r'##\w+', '', entity)
    return entity.lower().replace('-', ' ').strip()


def validate_entity_type(entity_type: str) -> str:
    """验证实体类型
    
    Args:
        entity_type: 实体类型
        
    Returns:
        str: 验证后的实体类型
    """
    valid_types = ['Food', 'Disease', 'Nutrient', 'DietaryPattern']
    if entity_type in valid_types:
        return entity_type
    
    type_mapping = {
        'food': 'Food',
        'disease': 'Disease',
        'nutrient': 'Nutrient',
        'dietarypattern': 'DietaryPattern',
        'dietary pattern': 'DietaryPattern'
    }
    return type_mapping.get(entity_type.lower(), 'Disease')


