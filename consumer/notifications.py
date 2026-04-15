def send_critical_alert(transaction_id: str, risk_score: int, rules: list):
    """
    Simulates sending an urgent notification via SMS/Email (e.g. Twilio, SendGrid) to Security Analysts.
    """
    print("\n" + "=" * 60)
    print("FRAUD ALERT DISPATCHED")
    print(f"Transaction ID : {transaction_id}")
    print(f"Risk Score     : {risk_score}/100")
    print(f"Triggered Rules: {', '.join(rules)}")
    print("=" * 60 + "\n")
