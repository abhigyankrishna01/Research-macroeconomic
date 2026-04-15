def calculate_risk_score(amount: float, velocity: int, is_ml_anomaly: bool) -> int:
    score = 0
    # Amount base rules
    if amount > 10000:
        score += 50
    elif amount > 5000:
        score += 25
    elif amount > 1000:
        score += 10

    # Velocity rules (Transactions per user in the last hour)
    if velocity > 100:
        score += 40
    elif velocity > 50:
        score += 20

    # Machine Learning Behavioral Context
    if is_ml_anomaly:
        score += 35

    return min(score, 100)
