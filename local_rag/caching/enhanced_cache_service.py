"""
Enhanced Cache Service - Redis-based caching for RAG performance optimization
Intelligent caching with TTL, compression, and cache warming
"""

import redis
import json
import hashlib
import pickle
import gzip
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import uuid
import asyncio
from dataclasses import dataclass, asdict
import logging

@dataclass
class CacheEntry:
    """
    Cache entry with metadata
    """
    key: str
    data: Any
    timestamp: datetime
    ttl: int
    hit_count: int = 0
    size_bytes: int = 0
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'ttl': self.ttl,
            'hit_count': self.hit_count,
            'size_bytes': self.size_bytes,
            'tags': self.tags or []
        }


class EnhancedCacheService:
    """
    Enhanced caching service with Redis backend and advanced features
    """
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None,
                 default_ttl: int = 3600,  # 1 hour
                 enable_compression: bool = True,
                 compression_threshold: int = 1024):  # 1KB
        """
        Initialize enhanced cache service
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            default_ttl: Default time-to-live in seconds
            enable_compression: Whether to compress large values
            compression_threshold: Size threshold for compression
        """
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # Cache prefixes for different data types
        self.prefixes = {
            'search': 'iabel:search:',
            'qa': 'iabel:qa:',
            'embeddings': 'iabel:emb:',
            'chunks': 'iabel:chunks:',
            'fusion': 'iabel:fusion:',
            'rerank': 'iabel:rerank:',
            'stats': 'iabel:stats:',
            'meta': 'iabel:meta:'
        }
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            print(f"✅ Redis cache connected: {redis_host}:{redis_port}")
            
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}")
            print("   Falling back to in-memory cache")
            self.redis_client = None
            self.connected = False
            self._memory_cache = {}
    
    def _create_key(self, prefix: str, identifier: str) -> str:
        """Create cache key with prefix"""
        return f"{prefix}{identifier}"
    
    def _hash_object(self, obj: Any) -> str:
        """Create hash from object for caching"""
        if isinstance(obj, str):
            content = obj
        elif isinstance(obj, dict):
            content = json.dumps(obj, sort_keys=True)
        else:
            content = str(obj)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression"""
        # Use pickle for serialization
        serialized = pickle.dumps(data)
        
        # Compress if enabled and data is large enough
        if (self.enable_compression and 
            len(serialized) > self.compression_threshold):
            serialized = gzip.compress(serialized)
            # Add compression marker
            serialized = b'GZIP:' + serialized
        
        return serialized
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data with decompression if needed"""
        try:
            # Check for compression marker
            if data.startswith(b'GZIP:'):
                data = gzip.decompress(data[5:])  # Remove marker
            
            return pickle.loads(data)
        except Exception as e:
            logging.error(f"Cache deserialization error: {e}")
            return None
    
    def set(self, 
            key: str, 
            value: Any, 
            ttl: Optional[int] = None,
            tags: Optional[List[str]] = None) -> bool:
        """
        Set cache value with TTL and tags
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            tags: Tags for cache invalidation
        """
        try:
            ttl = ttl or self.default_ttl
            
            if self.connected:
                # Redis backend
                serialized = self._serialize_data(value)
                
                # Set main value
                result = self.redis_client.setex(key, ttl, serialized)
                
                # Set tags for invalidation
                if tags:
                    for tag in tags:
                        tag_key = f"{self.prefixes['meta']}tag:{tag}"
                        self.redis_client.sadd(tag_key, key)
                        self.redis_client.expire(tag_key, ttl + 300)  # Tag lives longer
                
                # Update statistics
                self._update_cache_stats('set', len(serialized))
                
                return result
            else:
                # Memory fallback
                self._memory_cache[key] = {
                    'value': value,
                    'timestamp': datetime.now(),
                    'ttl': ttl,
                    'tags': tags or []
                }
                return True
                
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cache value
        
        Args:
            key: Cache key
        """
        try:
            if self.connected:
                # Redis backend
                data = self.redis_client.get(key)
                if data is None:
                    return None
                
                # Update hit statistics
                self._update_cache_stats('hit', 0)
                
                return self._deserialize_data(data)
            else:
                # Memory fallback
                entry = self._memory_cache.get(key)
                if entry is None:
                    return None
                
                # Check TTL
                elapsed = (datetime.now() - entry['timestamp']).total_seconds()
                if elapsed > entry['ttl']:
                    del self._memory_cache[key]
                    return None
                
                return entry['value']
                
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        try:
            if self.connected:
                return bool(self.redis_client.delete(key))
            else:
                return self._memory_cache.pop(key, None) is not None
        except Exception as e:
            logging.error(f"Cache delete error: {e}")
            return False
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """
        Invalidate cache entries by tags
        
        Args:
            tags: Tags to invalidate
            
        Returns:
            Number of keys invalidated
        """
        if not self.connected:
            # Memory fallback - simple implementation
            invalidated = 0
            keys_to_delete = []
            
            for key, entry in self._memory_cache.items():
                if any(tag in entry.get('tags', []) for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._memory_cache[key]
                invalidated += 1
            
            return invalidated
        
        try:
            invalidated = 0
            
            for tag in tags:
                tag_key = f"{self.prefixes['meta']}tag:{tag}"
                keys = self.redis_client.smembers(tag_key)
                
                if keys:
                    # Delete all keys with this tag
                    invalidated += self.redis_client.delete(*keys)
                    # Delete tag key
                    self.redis_client.delete(tag_key)
            
            return invalidated
            
        except Exception as e:
            logging.error(f"Cache invalidation error: {e}")
            return 0
    
    def cache_search_results(self, 
                           query: str, 
                           params: Dict[str, Any], 
                           results: List[Dict[str, Any]],
                           ttl: Optional[int] = None) -> bool:
        """Cache search results"""
        cache_key = self._create_search_key(query, params)
        tags = ['search', 'results', f"query:{self._hash_object(query)[:8]}"]
        return self.set(cache_key, results, ttl, tags)
    
    def get_search_results(self, 
                          query: str, 
                          params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        cache_key = self._create_search_key(query, params)
        return self.get(cache_key)
    
    def cache_qa_response(self, 
                         question: str, 
                         context_hash: str, 
                         response: Dict[str, Any],
                         ttl: Optional[int] = None) -> bool:
        """Cache QA response"""
        cache_key = self._create_qa_key(question, context_hash)
        tags = ['qa', 'response', f"question:{self._hash_object(question)[:8]}"]
        return self.set(cache_key, response, ttl, tags)
    
    def get_qa_response(self, 
                       question: str, 
                       context_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached QA response"""
        cache_key = self._create_qa_key(question, context_hash)
        return self.get(cache_key)
    
    def cache_embeddings(self, 
                        text_hash: str, 
                        embeddings: List[float],
                        ttl: int = 86400) -> bool:  # 24 hours for embeddings
        """Cache embeddings"""
        cache_key = self._create_key(self.prefixes['embeddings'], text_hash)
        tags = ['embeddings']
        return self.set(cache_key, embeddings, ttl, tags)
    
    def get_embeddings(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embeddings"""
        cache_key = self._create_key(self.prefixes['embeddings'], text_hash)
        return self.get(cache_key)
    
    def cache_fusion_results(self, 
                           query_hash: str, 
                           fusion_data: Dict[str, Any],
                           ttl: Optional[int] = None) -> bool:
        """Cache RAG fusion results"""
        cache_key = self._create_key(self.prefixes['fusion'], query_hash)
        tags = ['fusion', 'rag']
        return self.set(cache_key, fusion_data, ttl, tags)
    
    def get_fusion_results(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached fusion results"""
        cache_key = self._create_key(self.prefixes['fusion'], query_hash)
        return self.get(cache_key)
    
    def cache_rerank_results(self, 
                           query_docs_hash: str, 
                           reranked_results: List[Dict[str, Any]],
                           ttl: Optional[int] = None) -> bool:
        """Cache re-ranking results"""
        cache_key = self._create_key(self.prefixes['rerank'], query_docs_hash)
        tags = ['rerank', 'ranking']
        return self.set(cache_key, reranked_results, ttl, tags)
    
    def get_rerank_results(self, query_docs_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached re-ranking results"""
        cache_key = self._create_key(self.prefixes['rerank'], query_docs_hash)
        return self.get(cache_key)
    
    def _create_search_key(self, query: str, params: Dict[str, Any]) -> str:
        """Create cache key for search results"""
        key_data = {
            'query': query,
            'params': params
        }
        key_hash = self._hash_object(key_data)
        return self._create_key(self.prefixes['search'], key_hash)
    
    def _create_qa_key(self, question: str, context_hash: str) -> str:
        """Create cache key for QA response"""
        key_data = f"{question}:{context_hash}"
        key_hash = self._hash_object(key_data)
        return self._create_key(self.prefixes['qa'], key_hash)
    
    def _update_cache_stats(self, operation: str, size: int):
        """Update cache statistics"""
        if not self.connected:
            return
        
        try:
            stats_key = f"{self.prefixes['stats']}operations"
            pipeline = self.redis_client.pipeline()
            
            # Increment operation counter
            pipeline.hincrby(stats_key, operation, 1)
            
            # Update size statistics
            if operation == 'set':
                pipeline.hincrby(stats_key, 'total_size', size)
            
            # Set expiration
            pipeline.expire(stats_key, 86400)  # 24 hours
            
            pipeline.execute()
            
        except Exception as e:
            logging.error(f"Stats update error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.connected:
            return {
                'connected': False,
                'backend': 'memory',
                'entries': len(self._memory_cache)
            }
        
        try:
            # Get operation stats
            stats_key = f"{self.prefixes['stats']}operations"
            stats = self.redis_client.hgetall(stats_key)
            
            # Convert bytes to strings and integers
            processed_stats = {}
            for key, value in stats.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_int = int(value.decode()) if isinstance(value, bytes) else int(value)
                processed_stats[key_str] = value_int
            
            # Get Redis info
            redis_info = self.redis_client.info()
            
            return {
                'connected': True,
                'backend': 'redis',
                'operations': processed_stats,
                'redis_info': {
                    'used_memory': redis_info.get('used_memory_human', 'Unknown'),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0)
                },
                'cache_prefixes': list(self.prefixes.keys())
            }
            
        except Exception as e:
            logging.error(f"Stats retrieval error: {e}")
            return {
                'connected': False,
                'error': str(e)
            }
    
    def warm_cache(self, warm_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Warm cache with common queries and responses
        
        Args:
            warm_data: Dictionary with cache warming data
        """
        results = {
            'search_queries': 0,
            'qa_pairs': 0,
            'embeddings': 0,
            'errors': 0
        }
        
        try:
            # Warm search queries
            if 'search_queries' in warm_data:
                for query_data in warm_data['search_queries']:
                    success = self.cache_search_results(
                        query_data['query'],
                        query_data['params'],
                        query_data['results'],
                        ttl=7200  # 2 hours for warm data
                    )
                    if success:
                        results['search_queries'] += 1
                    else:
                        results['errors'] += 1
            
            # Warm QA pairs
            if 'qa_pairs' in warm_data:
                for qa_data in warm_data['qa_pairs']:
                    success = self.cache_qa_response(
                        qa_data['question'],
                        qa_data['context_hash'],
                        qa_data['response'],
                        ttl=7200
                    )
                    if success:
                        results['qa_pairs'] += 1
                    else:
                        results['errors'] += 1
            
            # Warm embeddings
            if 'embeddings' in warm_data:
                for emb_data in warm_data['embeddings']:
                    success = self.cache_embeddings(
                        emb_data['text_hash'],
                        emb_data['embeddings'],
                        ttl=86400  # 24 hours for embeddings
                    )
                    if success:
                        results['embeddings'] += 1
                    else:
                        results['errors'] += 1
            
            print(f"🔥 Cache warmed: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Cache warming error: {e}")
            results['errors'] += 1
            return results
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Args:
            pattern: Pattern to match keys (None = clear all IAbel cache)
        """
        try:
            if not self.connected:
                count = len(self._memory_cache)
                self._memory_cache.clear()
                return count
            
            # Default pattern for IAbel cache
            if pattern is None:
                pattern = "iabel:*"
            
            # Get matching keys
            keys = self.redis_client.keys(pattern)
            
            if keys:
                return self.redis_client.delete(*keys)
            else:
                return 0
                
        except Exception as e:
            logging.error(f"Cache clear error: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check"""
        try:
            if not self.connected:
                return {
                    'healthy': True,
                    'backend': 'memory',
                    'latency_ms': 0
                }
            
            # Test Redis connection with timing
            import time
            start_time = time.time()
            
            test_key = f"{self.prefixes['meta']}health_check"
            self.redis_client.set(test_key, 'test', ex=60)
            result = self.redis_client.get(test_key)
            self.redis_client.delete(test_key)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'healthy': result == b'test',
                'backend': 'redis',
                'latency_ms': round(latency_ms, 2)
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'backend': 'redis' if self.connected else 'memory',
                'error': str(e)
            }


# Global cache service instance
_global_cache_service = None

def get_enhanced_cache_service() -> EnhancedCacheService:
    """
    Get or create global enhanced cache service instance
    """
    global _global_cache_service
    if _global_cache_service is None:
        _global_cache_service = EnhancedCacheService()
    return _global_cache_service


# Backward compatibility
def get_cache_service() -> EnhancedCacheService:
    """Backward compatibility function"""
    return get_enhanced_cache_service()