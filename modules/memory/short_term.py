# modules/memory/short_term.py
import json
import redis
from datetime import timedelta
from typing import Optional, Any, Dict
from shared.settings import Settings


class SessionStore:
    """
    Lớp lưu trữ với hỗ trợ Redis cho short-term memory theo session_id.
    Được dùng chung cho recommendation caching.
    Fallback to in-memory storage if Redis is not available.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ttl = settings.redis_ttl
        self.redis = None
        self._memory_store: Dict[str, Any] = {}  # Fallback in-memory store
        
        # Try to connect to Redis
        try:
            self.redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            # Test connection
            self.redis.ping()
            print(f"✅ Connected to Redis at {settings.redis_url}")
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
            print("📦 Using in-memory storage as fallback")
            self.redis = None

    def get(self, session_id: str) -> Optional[dict]:
        if self.redis:
            try:
                raw = self.redis.get(session_id)
                if raw:
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return None
                return None
            except Exception as e:
                print(f"Redis get error: {e}")
                return self._memory_store.get(session_id)
        else:
            return self._memory_store.get(session_id)

    def set(self, session_id: str, data: Any, ttl_seconds: Optional[int] = None):
        """Set session data with optional TTL override."""
        ttl = ttl_seconds or self.ttl
        
        if self.redis:
            try:
                self.redis.set(session_id, json.dumps(data, default=str), ex=ttl)
                return
            except Exception as e:
                print(f"Redis set error: {e}, falling back to memory")
        
        # Fallback to memory store
        self._memory_store[session_id] = data

    def delete(self, session_id: str):
        if self.redis:
            try:
                self.redis.delete(session_id)
            except Exception as e:
                print(f"Redis delete error: {e}")
        
        # Also delete from memory store
        self._memory_store.pop(session_id, None)

    def update(self, session_id: str, key: str, value: Any):
        state = self.get(session_id) or {}
        state[key] = value
        self.set(session_id, state)

    def clear(self):
        """WARNING: Clears all keys – use only in development/testing."""
        if self.redis:
            try:
                self.redis.flushdb()  # Chỉ nên dùng trong test/demo
            except Exception as e:
                print(f"Redis clear error: {e}")
        
        # Clear memory store
        self._memory_store.clear() 