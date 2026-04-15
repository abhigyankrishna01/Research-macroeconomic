import os
import json
import asyncio
import io
from datetime import datetime
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from confluent_kafka import Consumer, KafkaError
import pandas as pd
import mlflow.sklearn

from models import SessionLocal, Transaction, Alert, AuditLog
from enrichment import update_user_features, increment_malicious_tally, get_malicious_tally
from scoring import calculate_risk_score
from notifications import send_critical_alert

app = FastAPI(title="Fraud Detection Engine")

# Add CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
TOPIC = "transactions_in"
GROUP_ID = "fraud_detection_group"

consumer_conf = {
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': GROUP_ID,
    'auto.offset.reset': 'earliest'
}

ml_model = None
ml_model_mode = "none"
ws_clients = []

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_clients.remove(websocket)

async def broadcast_alert(data: dict):
    for client in ws_clients:
        try:
            await client.send_json(data)
        except Exception:
            pass


def load_ml_model():
    global ml_model, ml_model_mode
    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", "http://localhost:5000"))
    try:
        model_uri = "models:/fraud_classifier/latest"
        print(f"Loading ML model from {model_uri}...")
        ml_model = mlflow.sklearn.load_model(model_uri)
        ml_model_mode = "supervised"
        print("Supervised ML model loaded successfully.")
    except Exception as supervised_error:
        try:
            model_uri = "models:/fraud_iforest/latest"
            print(f"Loading fallback ML model from {model_uri}...")
            ml_model = mlflow.sklearn.load_model(model_uri)
            ml_model_mode = "unsupervised"
            print("Fallback unsupervised ML model loaded successfully.")
        except Exception as iforest_error:
            ml_model = None
            ml_model_mode = "none"
            print(
                "Warning: Could not load ML model. ML scoring is disabled. "
                f"Supervised error: {supervised_error}; fallback error: {iforest_error}"
            )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def evaluate_transaction(transaction: dict, user_tx_count: int):
    amount = float(transaction.get('amount', 0))
    ts = datetime.fromisoformat(transaction['timestamp'])

    rules_triggered = []

    # 1. Base Rules Check
    if amount > 10000:
        rules_triggered.append("HIGH_AMOUNT_THRESHOLD")

    if user_tx_count > 100:
        rules_triggered.append("HIGH_VELOCITY")

    # 2. ML Prediction Check
    is_ml_anomaly = False
    if ml_model:
        df = pd.DataFrame([{
            'amount': amount,
            'merchant_category': transaction['merchant_category'],
            'hour': ts.hour
        }])
        pred = ml_model.predict(df)[0]
        if (ml_model_mode == "unsupervised" and pred == -1) or (ml_model_mode == "supervised" and pred == 1):
            rules_triggered.append("ML_BEHAVIORAL_ANOMALY")
            is_ml_anomaly = True

    # 3. Aggregate composite risk score
    risk_score = calculate_risk_score(amount, user_tx_count, is_ml_anomaly)

    return risk_score, rules_triggered

def process_message(msg_value: str, db: Session):
    try:
        data = json.loads(msg_value)
        ts = datetime.fromisoformat(data['timestamp'])

        user_tx_count = update_user_features(data['user_id'], ts)

        risk_score, rules_triggered = evaluate_transaction(data, user_tx_count)

        # Insert Transaction
        tx = Transaction(
            transaction_id=data['transaction_id'],
            user_id=data['user_id'],
            amount=data['amount'],
            merchant_category=data['merchant_category'],
            location=data['location'],
            timestamp=ts,
            status='PROCESSED',
            risk_score=risk_score
        )
        db.add(tx)
        db.flush()  # Secure Transaction object into the DB sequentially

        # Queue actionable alerts based on risk tier
        malicious_tally = get_malicious_tally(data['user_id'])
        if risk_score > 40:
            severity = "CRITICAL" if risk_score >= 80 else "MEDIUM"
            rule_string = ",".join(rules_triggered) if rules_triggered else "COMPOSITE_RISK"

            alert = Alert(
                transaction_id=data['transaction_id'],
                rule_triggered=rule_string,
                severity=severity,
                created_at=datetime.utcnow()
            )
            db.add(alert)

            if risk_score >= 80:
                # Dispatch real-time emergency out-of-band notification
                send_critical_alert(data['transaction_id'], risk_score, rules_triggered)
                # 3-strike policy: hard block only after sustained malicious activity.
                malicious_tally = increment_malicious_tally(data['user_id'])
                if malicious_tally >= 3:
                    tx.status = 'BLOCKED'

        # Save overarching audit log
        log_details = f"Processed TX. Score: {risk_score}. Velocity: {user_tx_count}."
        audit = AuditLog(
            action="TRANSACTION_PROCESSED",
            entity_id=data['transaction_id'],
            details=log_details,
            timestamp=datetime.utcnow()
        )
        db.add(audit)

        db.commit()

        # Return object for websocket broadcasting
        data['risk_score'] = risk_score
        data['rules_triggered'] = rules_triggered
        data['status'] = tx.status
        data['malicious_tally'] = malicious_tally
        return data

    except Exception as e:
        print(f"Error processing message: {e}")
        db.rollback()
        return None

async def consume_loop():
    consumer = Consumer(consumer_conf)
    consumer.subscribe([TOPIC])
    print(f"Subscribed to topic {TOPIC}")
    db = SessionLocal()
    try:
        while True:
            await asyncio.sleep(0.1)
            msg = consumer.poll(0.1)

            if msg is None: continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print("Kafka Consumer Error:", msg.error())
                    await asyncio.sleep(1)
                    continue

            payload = process_message(msg.value().decode('utf-8'), db)
            if payload:
                asyncio.create_task(broadcast_alert(payload))
    finally:
        consumer.close()
        db.close()

@app.on_event("startup")
async def startup_event():
    load_ml_model()
    asyncio.create_task(consume_loop())

@app.get("/")
def read_root():
    return {"status": "Fraud Detection Engine is running in Phase 3 mode"}


class TransactionLabelPayload(BaseModel):
    is_fraud: bool


@app.patch("/api/transactions/{tx_id}/label")
def label_transaction(
    tx_id: str,
    payload: TransactionLabelPayload,
    db: Session = Depends(get_db)
):
    tx = None
    if tx_id.isdigit():
        tx = db.query(Transaction).filter(Transaction.id == int(tx_id)).first()
    if tx is None:
        tx = db.query(Transaction).filter(Transaction.transaction_id == tx_id).first()
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    tx.is_fraud = payload.is_fraud
    db.add(
        AuditLog(
            action="TRANSACTION_LABELED",
            entity_id=tx.transaction_id,
            details=f"Analyst label set is_fraud={payload.is_fraud}",
            timestamp=datetime.utcnow()
        )
    )
    db.commit()
    db.refresh(tx)
    return {
        "id": tx.id,
        "transaction_id": tx.transaction_id,
        "is_fraud": tx.is_fraud,
        "status": tx.status
    }


@app.get("/api/transactions/recent")
def recent_transactions(limit: int = Query(50), db: Session = Depends(get_db)):
    rows = (
        db.query(Transaction)
        .order_by(Transaction.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "transaction_id": t.transaction_id,
            "user_id": t.user_id,
            "amount": float(t.amount),
            "merchant_category": t.merchant_category,
            "location": t.location,
            "timestamp": t.timestamp.isoformat(),
            "risk_score": t.risk_score or 0,
            "status": t.status,
            "is_fraud": t.is_fraud,
            "rules_triggered": [],
            "malicious_tally": 0,
        }
        for t in reversed(rows)
    ]


@app.get("/api/transactions/recent/critical")
def recent_critical(limit: int = Query(50), db: Session = Depends(get_db)):
    rows = (
        db.query(Transaction)
        .filter(Transaction.risk_score >= 80)
        .order_by(Transaction.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "transaction_id": t.transaction_id,
            "user_id": t.user_id,
            "amount": float(t.amount),
            "merchant_category": t.merchant_category,
            "location": t.location,
            "timestamp": t.timestamp.isoformat(),
            "risk_score": t.risk_score or 0,
            "status": t.status,
            "is_fraud": t.is_fraud,
            "rules_triggered": ["HIGH_AMOUNT_THRESHOLD", "HIGH_VELOCITY"],
            "malicious_tally": 0,
        }
        for t in reversed(rows)
    ]


@app.get("/api/reports/summary")
def report_summary(
    download: bool = Query(False, description="When true, downloads labeled dataset CSV"),
    db: Session = Depends(get_db)
):
    total_flagged = db.query(func.count(Transaction.id)).filter(Transaction.risk_score > 40).scalar() or 0
    auto_blocked_accounts = db.query(func.count(func.distinct(Transaction.user_id))).filter(
        Transaction.status == "BLOCKED"
    ).scalar() or 0
    verified_frauds = db.query(func.count(Transaction.id)).filter(Transaction.is_fraud.is_(True)).scalar() or 0

    if download:
        rows = db.query(
            Transaction.transaction_id,
            Transaction.user_id,
            Transaction.amount,
            Transaction.merchant_category,
            Transaction.location,
            Transaction.timestamp,
            Transaction.status,
            Transaction.risk_score,
            Transaction.is_fraud,
        ).filter(Transaction.is_fraud.is_not(None)).all()
        report_df = pd.DataFrame(
            rows,
            columns=[
                "transaction_id",
                "user_id",
                "amount",
                "merchant_category",
                "location",
                "timestamp",
                "status",
                "risk_score",
                "is_fraud"
            ]
        )
        csv_buffer = io.StringIO()
        report_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        filename = f"fraud-report-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    return {
        "total_flagged": total_flagged,
        "auto_blocked_accounts": auto_blocked_accounts,
        "verified_frauds": verified_frauds
    }
