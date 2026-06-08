#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index.py - 构建 FAISS 向量索引（只运行一次）

功能：
- 读取食物营养素本体和疾病本体
- 对食物和营养素实体去重并编码
- 构建 FAISS IndexFlatIP 索引
- 持久化保存索引文件供 grounding_alignment.py 使用

按照 BUILD_INDEX_SPEC.md v1.0 实现
"""

import os
import pickle
import faiss
import pandas as pd
import numpy as np
from settings import (
    NUTRIENT_TRIPLETS_FILE,
    HUMANIDO_FILE,
    INDEX_DIR,
    FOOD_MODEL,
    SAPBERT_BATCH,
    BUILD_FOOD_INDEX,
    BUILD_NUTRIENT_INDEX
)


def main():
    """主函数"""
    print("=" * 70)
    print("build_index.py - 构建 FAISS 向量索引")
    print("=" * 70)
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"索引目录: {INDEX_DIR}")
    
    food_index_path = os.path.join(INDEX_DIR, "food_desc.faiss")
    nutrient_index_path = os.path.join(INDEX_DIR, "nutrient_obj.faiss")
    
    if BUILD_FOOD_INDEX and os.path.exists(food_index_path):
        print("食物索引已存在，跳过构建")
    elif BUILD_FOOD_INDEX or BUILD_NUTRIENT_INDEX:
        print(f"\n读取本体文件: {NUTRIENT_TRIPLETS_FILE}")
        df = pd.read_csv(NUTRIENT_TRIPLETS_FILE, dtype=str)
        print(f"成功读取 {len(df)} 条记录")
        
        if BUILD_FOOD_INDEX and not os.path.exists(food_index_path):
            build_food_index(df)
        
        if BUILD_NUTRIENT_INDEX and not os.path.exists(nutrient_index_path):
            build_nutrient_index(df)
    
    print("\n" + "=" * 70)
    print("索引构建完成！")
    print("=" * 70)


def build_food_index(df: pd.DataFrame):
    """构建食物索引"""
    print("\n" + "-" * 70)
    print("构建食物索引...")
    print("-" * 70)
    
    df_food = df.drop_duplicates(subset=["description"], keep="first")
    desc_list = df_food["description"].tolist()
    print(f"食物去重后条目数: {len(desc_list)}")
    
    attr_list = []
    for _, row in df_food.iterrows():
        attr_list.append({
            "fdc_id": row.get("fdc_id") if pd.notna(row.get("fdc_id")) else None,
            "full_portion": row.get("full_portion") if pd.notna(row.get("full_portion")) else None,
            "gram_weight": row.get("gram_weight") if pd.notna(row.get("gram_weight")) else None,
            "food_attributes": row.get("food_attributes") if pd.notna(row.get("food_attributes")) else None
        })
    
    print("加载 SentenceTransformer 模型...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(FOOD_MODEL)
    
    print("编码食物描述...")
    embeddings = model.encode(
        desc_list,
        batch_size=SAPBERT_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    faiss.write_index(index, os.path.join(INDEX_DIR, "food_desc.faiss"))
    with open(os.path.join(INDEX_DIR, "food_desc.pkl"), "wb") as f:
        pickle.dump(desc_list, f)
    with open(os.path.join(INDEX_DIR, "food_attr.pkl"), "wb") as f:
        pickle.dump(attr_list, f)
    
    print(f"食物索引构建完成，共 {len(desc_list)} 条")


def build_nutrient_index(df: pd.DataFrame):
    """构建营养素索引"""
    print("\n" + "-" * 70)
    print("构建营养素索引...")
    print("-" * 70)
    
    df_nutrient = df.drop_duplicates(subset=["object"], keep="first")
    nutrient_list = df_nutrient["object"].tolist()
    print(f"营养素去重后条目数: {len(nutrient_list)}")
    
    nutrient_attr_list = []
    for _, row in df_nutrient.iterrows():
        nutrient_attr_list.append({
            "nutrient_id": row.get("nutrient_id") if pd.notna(row.get("nutrient_id")) else None,
            "nutrient_number": row.get("nutrient_number") if pd.notna(row.get("nutrient_number")) else None,
            "unit_name": row.get("unit_name") if pd.notna(row.get("unit_name")) else None,
            "amount": row.get("amount") if pd.notna(row.get("amount")) else None
        })
    
    print("加载 SentenceTransformer 模型...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(FOOD_MODEL)
    
    print("编码营养素名称...")
    embeddings = model.encode(
        nutrient_list,
        batch_size=SAPBERT_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    faiss.write_index(index, os.path.join(INDEX_DIR, "nutrient_obj.faiss"))
    with open(os.path.join(INDEX_DIR, "nutrient_obj.pkl"), "wb") as f:
        pickle.dump(nutrient_list, f)
    with open(os.path.join(INDEX_DIR, "nutrient_attr.pkl"), "wb") as f:
        pickle.dump(nutrient_attr_list, f)
    
    print(f"营养素索引构建完成，共 {len(nutrient_list)} 条")


if __name__ == "__main__":
    main()


