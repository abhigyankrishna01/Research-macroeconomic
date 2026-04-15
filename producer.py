import os
import json
import time
import random
import uuid
from datetime import datetime
from confluent_kafka import Producer
from faker import Faker

fake = Faker()

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
TOPIC = "transactions_in"

producer_conf = {
    'bootstrap.servers': KAFKA_BROKER
}

producer = Producer(producer_conf)

MERCHANT_CATEGORIES = [
    "retail", "groceries", "electronics", "online_subscriptions",
    "travel", "restaurants", "gas_station", "entertainment"
]

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def generate_transaction():
    amount = round(random.uniform(5.0, 5000.0), 2)
    # create some anomalies: high amounts
    if random.random() < 0.05:
        amount = round(random.uniform(10000.0, 50000.0), 2)

    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "amount": amount,
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "location": f"{fake.city()}, {fake.country()}",
        "timestamp": datetime.utcnow().isoformat()
    }
    return transaction

def main():
    print(f"Starting transaction producer, broker={KAFKA_BROKER}, topic={TOPIC}")
    try:
        while True:
            transaction = generate_transaction()
            producer.produce(
                TOPIC,
                key=transaction['user_id'],
                value=json.dumps(transaction),
                callback=delivery_report
            )
            producer.poll(0)

            # simulate 2 to 10 transactions per second
            time.sleep(random.uniform(0.1, 0.5))
    except KeyboardInterrupt:
        print("Stopping producer...")
    except Exception as e:
        print(f"Producer crashed: {e}")
    finally:
        producer.flush()

if __name__ == "__main__":
    main()
