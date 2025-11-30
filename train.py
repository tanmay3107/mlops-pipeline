import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_wine
import mlflow
import mlflow.sklearn

# 1. HELPER: Calculate Metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 2. LOAD DATA (Simulating an Ingestion Pipeline)
    print("Loading data...")
    data = load_wine(as_frame=True)
    df = data.frame
    
    # We'll predict 'alcohol' content based on other chemical properties
    # (Just a dummy regression task for this demo)
    X = df.drop(["alcohol"], axis=1)
    y = df["alcohol"]

    # Split
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=40)

    # 3. HYPERPARAMETERS (We read these from command line for automation later)
    # If user doesn't provide them, default to 0.5
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # 4. MLFLOW TRACKING (The "MLOps" Magic)
    # This block logs everything to a local server
    print(f"Training with alpha={alpha}, l1_ratio={l1_ratio}...")
    
    with mlflow.start_run():
        # A. Train Model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # B. Predict
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"  RMSE: {rmse}")
        print(f"  R2: {r2}")

        # C. Log Parameters & Metrics (Crucial for Dashboard)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # D. Log the actual Model file
        mlflow.sklearn.log_model(lr, "model")
        print("✅ Model saved to MLflow.")