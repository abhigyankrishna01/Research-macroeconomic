import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5433/fraud_db")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
SUPERVISED_LABEL_THRESHOLD = int(os.getenv("SUPERVISED_LABEL_THRESHOLD", "50"))

def load_all_data():
    print("Loading data from database...")
    engine = create_engine(DATABASE_URL)
    query = "SELECT amount, merchant_category, timestamp, is_fraud FROM transactions"
    df = pd.read_sql(query, engine)

    # Feature Engineering
    if not df.empty:
        # Extract hour from timestamp
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df.drop(columns=['timestamp'], inplace=True)
    return df


def build_preprocessor():
    numeric_features = ['amount', 'hour']
    categorical_features = ['merchant_category']
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

def train_and_log_model():
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment("Fraud_Anomaly_Detection")

    df = load_all_data()
    if df.empty:
        print("No training data available. Run producer for a while.")
        return

    labeled_df = df[df['is_fraud'].notna()].copy()
    print(f"Total transactions available: {len(df)}. Labeled transactions: {len(labeled_df)}.")

    preprocessor = build_preprocessor()

    using_supervised = len(labeled_df) >= SUPERVISED_LABEL_THRESHOLD
    if using_supervised:
        print("Threshold reached. Training supervised RandomForestClassifier.")
        X = labeled_df[['amount', 'merchant_category', 'hour']]
        y = labeled_df['is_fraud'].astype(int)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        registered_model_name = "fraud_classifier"
        artifact_path = "random_forest_classifier"
    else:
        print(
            f"Insufficient labels (< {SUPERVISED_LABEL_THRESHOLD}). "
            "Falling back to unsupervised IsolationForest."
        )
        X = df[['amount', 'merchant_category', 'hour']]
        y = None
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', IsolationForest(contamination=0.05, random_state=42))
        ])
        registered_model_name = "fraud_iforest"
        artifact_path = "isolation_forest"

    with mlflow.start_run() as run:
        if using_supervised:
            model.fit(X, y)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("labeled_samples", len(labeled_df))
            mlflow.log_param("label_threshold", SUPERVISED_LABEL_THRESHOLD)
        else:
            model.fit(X)
            mlflow.log_param("model_type", "IsolationForest")
            mlflow.log_param("labeled_samples", len(labeled_df))
            mlflow.log_param("label_threshold", SUPERVISED_LABEL_THRESHOLD)

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        print(f"Model logged in run {run.info.run_id}")

if __name__ == "__main__":
    train_and_log_model()
