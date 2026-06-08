#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 调用客户端

功能：
- 封装 API 调用
- 指数退避重试机制
- 缓存功能（基于 MD5 哈希）
- 调用统计
"""

import logging
import hashlib
import time
import json
from typing import Dict, Optional
from pathlib import Path

from settings import (
    API_KEY, API_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    API_RETRY_TIMES, API_RETRY_INTERVAL, BETA_DECAY,
    API_RATE_LIMIT_WAIT_1, API_RATE_LIMIT_WAIT_2, API_RATE_LIMIT_WAIT_3,
    API_SKIP_AFTER_WAIT
)

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM gateway used by DIETA agents.

    This class is the single model-call gateway for the agentized release:
    it centralizes retry, cache, rate-limit handling and call statistics,
    making LLM access auditable rather than scattered across agent modules.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        初始化 LLM 客户端

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir or Path(__file__).parent / "results" / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache: Dict[str, str] = {}
        self.cache_file = self.cache_dir / "llm_cache.json"

        self._load_cache()

        self.call_count = 0
        self.cache_hit_count = 0

    def _load_cache(self):
        """加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"加载了 {len(self.cache)} 条缓存")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self.cache = {}

    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _get_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """
        生成缓存键

        Args:
            prompt: 提示词
            model: 模型名称
            temperature: 温度参数

        Returns:
            str: 缓存键
        """
        key_str = f"{prompt}|{model}|{temperature}"
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    def generate(self, prompt: str,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 use_cache: bool = True) -> str:
        """
        生成 LLM 响应

        Args:
            prompt: 提示词
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            use_cache: 是否使用缓存

        Returns:
            str: LLM 响应
        """
        model = model or LLM_MODEL
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        max_tokens = max_tokens or LLM_MAX_TOKENS

        self.call_count += 1

        if use_cache:
            cache_key = self._get_cache_key(prompt, model, temperature)
            if cache_key in self.cache:
                self.cache_hit_count += 1
                return self.cache[cache_key]

        for attempt in range(API_RETRY_TIMES):
            try:
                response = self._call_api(prompt, model, temperature, max_tokens)

                if use_cache:
                    cache_key = self._get_cache_key(prompt, model, temperature)
                    self.cache[cache_key] = response
                    self._save_cache()

                return response

            except Exception as e:
                if attempt < API_RETRY_TIMES - 1:
                    wait_time = API_RETRY_INTERVAL * (BETA_DECAY ** attempt)
                    logger.warning(f"API 调用失败 (尝试 {attempt + 1}/{API_RETRY_TIMES}): {e}")
                    logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API 调用最终失败: {e}")
                    raise

    def _call_api(self, prompt: str, model: str,
                  temperature: float, max_tokens: int,
                  rate_limit_attempt: int = 0) -> str:
        """
        调用 LLM API

        Args:
            prompt: 提示词
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            rate_limit_attempt: 当前限流尝试次数

        Returns:
            str: API 响应
        """
        logger.info(f"调用 LLM API: model={model}, temperature={temperature}")

        try:
            import requests

            url = f"{API_BASE_URL}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }

            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            # 处理API限流
            if response.status_code == 429:
                rate_limit_attempt += 1
                logger.warning(f"API限流 (尝试 {rate_limit_attempt}/3)")
                
                # 优先使用Retry-After头
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                    logger.info(f"API返回Retry-After: {wait_time}秒")
                else:
                    # 使用指数退避策略
                    if rate_limit_attempt == 1:
                        wait_time = API_RATE_LIMIT_WAIT_1
                    elif rate_limit_attempt == 2:
                        wait_time = API_RATE_LIMIT_WAIT_2
                    else:
                        wait_time = API_RATE_LIMIT_WAIT_3
                
                if rate_limit_attempt <= 3:
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    return self._call_api(prompt, model, temperature, max_tokens, rate_limit_attempt)
                else:
                    if API_SKIP_AFTER_WAIT:
                        logger.error(f"已达到最大限流等待次数，跳过当前记录")
                        raise Exception("API限流达到最大等待次数")
                    else:
                        raise
            
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except ImportError as e:
            raise RuntimeError("The requests package is required for model API calls.") from e
        except Exception as e:
            if "API限流达到最大等待次数" in str(e):
                raise
            raise RuntimeError(f"Model API call failed: {e}") from e

    def get_stats(self) -> Dict:
        """
        获取调用统计

        Returns:
            Dict: 统计信息
        """
        cache_hit_rate = self.cache_hit_count / self.call_count if self.call_count > 0 else 0
        return {
            "total_calls": self.call_count,
            "cache_hits": self.cache_hit_count,
            "cache_misses": self.call_count - self.cache_hit_count,
            "cache_hit_rate": cache_hit_rate
        }




