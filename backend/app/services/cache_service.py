"""
Redis Cache Service for Enhanced RAG Performance
Caches embeddings, search results, and frequently accessed data
"""

import redis
import json
import hashlib
import io
import numpy as np
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from collections import OrderedDict
import os
from functools import wraps
import asyncio
import logging

logger = logging.getLogger(__name__)

# Byte prefix used to distinguish serialized numpy data from JSON
_NUMPY_MAGIC = b'\x93NUMPY'


class CacheService:
    """
    Redis-based caching service for RAG system performance optimization
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 default_ttl: int = 3600,  # 1 hour
                 embedding_ttl: int = 86400,  # 24 hours
                 search_ttl: int = 1800):  # 30 minutes
        """
        Initialize cache service
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default cache TTL in seconds
            embedding_ttl: Embedding cache TTL in seconds
            search_ttl: Search results cache TTL in seconds
        """
        self.default_ttl = default_ttl
        self.embedding_ttl = embedding_ttl
        self.search_ttl = search_ttl
        
        self._memory_maxsize = 1000  # max keys for in-memory LRU cache

        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis cache service connected")
        except (redis.ConnectionError, redis.RedisError) as e:
            logger.warning("Redis not available: %s — using in-memory cache", e)
            self.redis_available = False
            self.memory_cache: OrderedDict = OrderedDict()   # LRU order
            self.cache_timestamps: dict = {}
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """
        Generate a consistent cache key from data
        
        Args:
            prefix: Key prefix (e.g., 'embed', 'search')
            data: Data to hash for key generation
        """
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        hash_obj = hashlib.md5(data_str.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage. Uses numpy format for arrays, JSON for everything else."""
        if isinstance(data, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, data)
            return buf.getvalue()
        if isinstance(data, list) and data and isinstance(data[0], np.ndarray):
            buf = io.BytesIO()
            np.save(buf, np.array(data))
            return buf.getvalue()
        try:
            return json.dumps(data).encode('utf-8')
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot serialize cache value: {e}") from e

    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage. Never uses pickle."""
        if data[:6] == _NUMPY_MAGIC:
            buf = io.BytesIO(data)
            return np.load(buf, allow_pickle=False)
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Cannot deserialize cache data: {e}") from e
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        try:
            ttl = ttl or self.default_ttl
            serialized_data = self._serialize_data(value)
            
            if self.redis_available:
                self.redis_client.setex(key, ttl, serialized_data)
            else:
                # LRU eviction: remove oldest entry when at capacity
                if key not in self.memory_cache and len(self.memory_cache) >= self._memory_maxsize:
                    oldest_key, _ = self.memory_cache.popitem(last=False)
                    self.cache_timestamps.pop(oldest_key, None)
                # Move to end (most recently used)
                self.memory_cache[key] = serialized_data
                self.memory_cache.move_to_end(key)
                self.cache_timestamps[key] = datetime.now() + timedelta(seconds=ttl)

            return True
        except Exception as e:
            logger.warning("Cache set error: %s", e)
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
        """
        try:
            if self.redis_available:
                data = self.redis_client.get(key)
                if data is not None:
                    return self._deserialize_data(data)
            else:
                # In-memory fallback
                if key in self.memory_cache:
                    # Check expiration
                    if datetime.now() < self.cache_timestamps[key]:
                        return self._deserialize_data(self.memory_cache[key])
                    else:
                        # Remove expired key
                        del self.memory_cache[key]
                        del self.cache_timestamps[key]
            
            return None
        except Exception as e:
            logger.warning("Cache get error: %s", e)
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from cache
        """
        try:
            if self.redis_available:
                self.redis_client.delete(key)
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    del self.cache_timestamps[key]
            return True
        except Exception as e:
            logger.warning("Cache delete error: %s", e)
            return False
    
    def cache_embedding(self, text: str, embedding: np.ndarray) -> bool:
        """
        Cache text embedding
        
        Args:
            text: Source text
            embedding: Embedding vector
        """
        key = self._generate_cache_key("embed", text)
        return self.set(key, embedding, self.embedding_ttl)
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for text
        
        Args:
            text: Source text
        """
        key = self._generate_cache_key("embed", text)
        return self.get(key)
    
    def cache_search_results(self, query: str, params: Dict[str, Any], results: List[Dict[str, Any]]) -> bool:
        """
        Cache search results
        
        Args:
            query: Search query
            params: Search parameters (top_k, threshold, etc.)
            results: Search results
        """
        cache_data = {"query": query, "params": params}
        key = self._generate_cache_key("search", cache_data)
        return self.set(key, results, self.search_ttl)
    
    def get_search_results(self, query: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search results
        
        Args:
            query: Search query
            params: Search parameters
        """
        cache_data = {"query": query, "params": params}
        key = self._generate_cache_key("search", cache_data)
        return self.get(key)
    
    def cache_qa_response(self, question: str, context_hash: str, response: Dict[str, Any]) -> bool:
        """
        Cache QA response
        
        Args:
            question: User question
            context_hash: Hash of context documents
            response: Generated response
        """
        cache_data = {"question": question, "context_hash": context_hash}
        key = self._generate_cache_key("qa", cache_data)
        return self.set(key, response, self.search_ttl)
    
    def get_qa_response(self, question: str, context_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached QA response
        
        Args:
            question: User question
            context_hash: Hash of context documents
        """
        cache_data = {"question": question, "context_hash": context_hash}
        key = self._generate_cache_key("qa", cache_data)
        return self.get(key)
    
    def clear_cache(self, pattern: Optional[str] = None) -> bool:
        """
        Clear cache entries
        
        Args:
            pattern: Key pattern to clear (None for all)
        """
        try:
            if self.redis_available:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                else:
                    self.redis_client.flushdb()
            else:
                if pattern:
                    # Simple pattern matching for in-memory cache
                    keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
                    for key in keys_to_delete:
                        del self.memory_cache[key]
                        if key in self.cache_timestamps:
                            del self.cache_timestamps[key]
                else:
                    self.memory_cache.clear()
                    self.cache_timestamps.clear()
            
            return True
        except Exception as e:
            logger.warning("Cache clear error: %s", e)
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        try:
            if self.redis_available:
                info = self.redis_client.info()
                return {
                    "cache_type": "Redis",
                    "connected": True,
                    "used_memory": info.get("used_memory_human", "Unknown"),
                    "total_keys": self.redis_client.dbsize(),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
            else:
                # Clean expired entries first
                now = datetime.now()
                expired_keys = [k for k, exp_time in self.cache_timestamps.items() if now >= exp_time]
                for key in expired_keys:
                    del self.memory_cache[key]
                    del self.cache_timestamps[key]
                
                return {
                    "cache_type": "In-Memory",
                    "connected": True,
                    "total_keys": len(self.memory_cache),
                    "memory_usage": "Unknown"
                }
        except Exception as e:
            return {
                "cache_type": "Redis" if self.redis_available else "In-Memory",
                "connected": False,
                "error": str(e)
            }


def cache_decorator(cache_service: CacheService, prefix: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results
    
    Args:
        cache_service: Cache service instance
        prefix: Cache key prefix
        ttl: Cache TTL in seconds
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = cache_service._generate_cache_key(prefix, {"args": args, "kwargs": kwargs})
            
            # Try to get from cache first
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_service.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = cache_service._generate_cache_key(prefix, {"args": args, "kwargs": kwargs})
            
            # Try to get from cache first
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_service.set(cache_key, result, ttl)
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global cache service instance
cache_service = None

def get_cache_service() -> CacheService:
    """
    Get or create the cache service instance
    """
    global cache_service
    if cache_service is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        cache_service = CacheService(redis_url=redis_url)
    return cache_service