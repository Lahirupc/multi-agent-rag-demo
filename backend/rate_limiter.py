import time
import asyncio
from fastapi import HTTPException
from logger import logger

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate # Tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_refill
            new_tokens = time_passed * self.refill_rate
            
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

# Manage rate limits per user (In a real system, you'd use Redis for this)
class RateLimiterManager:
    def __init__(self, default_capacity=10, default_refill_rate=1.0):
        self.buckets = {}
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self.lock = asyncio.Lock()

    async def get_bucket(self, user_id: str) -> TokenBucket:
        async with self.lock:
            if user_id not in self.buckets:
                self.buckets[user_id] = TokenBucket(self.default_capacity, self.default_refill_rate)
            return self.buckets[user_id]

    async def check_rate_limit(self, user_id: str):
        bucket = await self.get_bucket(user_id)
        if not await bucket.consume(1):
            logger.warning("rate_limit_exceeded", user_id=user_id)
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

# Global rate limiter manager object
rate_limiter = RateLimiterManager(default_capacity=5, default_refill_rate=0.5) # 5 requests burst, 1 request per 2 secs
