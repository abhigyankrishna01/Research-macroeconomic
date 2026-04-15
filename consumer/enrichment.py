import redis
import os
from datetime import datetime

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Using decode_responses=True stringifies bytes
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def update_user_features(user_id: str, timestamp: datetime):
    """
    Increments the number of transactions by user in the current hour.
    """
    hour_key = timestamp.strftime("%Y-%m-%d-%H")
    redis_key = f"user_tx_count:{user_id}:{hour_key}"

    count = redis_client.incr(redis_key)
    # Expire in 25 hours
    if count == 1:
        redis_client.expire(redis_key, 25 * 3600)

    return count


def increment_malicious_tally(user_id: str):
    """
    Increment long-lived high-risk tally used for 3-strike blocking.
    """
    redis_key = f"malicious_tally:{user_id}"
    tally = redis_client.incr(redis_key)
    if tally == 1:
        redis_client.expire(redis_key, 30 * 24 * 3600)
    return tally


def get_malicious_tally(user_id: str):
    redis_key = f"malicious_tally:{user_id}"
    tally = redis_client.get(redis_key)
    return int(tally) if tally is not None else 0
