CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    merchant_category VARCHAR(255),
    location VARCHAR(255),
    timestamp TIMESTAMP NOT NULL,
    status VARCHAR(50) DEFAULT 'PENDING'
);

ALTER TABLE transactions
ADD COLUMN IF NOT EXISTS is_fraud BOOLEAN DEFAULT NULL;

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) REFERENCES transactions(transaction_id),
    rule_triggered VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    action VARCHAR(255) NOT NULL,
    entity_id VARCHAR(255),
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
