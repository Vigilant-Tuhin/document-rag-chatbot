import os
import redis

# docker-compose sets REDIS_URL=redis://redis:6379; fall back to that same
# host for local (non-Docker) runs where the env var won't be set.
# REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379").strip()

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

if __name__ == "__main__":
    print("Redis connection:", redis_client.ping())
