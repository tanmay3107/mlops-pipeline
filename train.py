import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_wine
import mlflow
import mlflow.sklearn

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1. READ ARGUMENTS
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    # NEW: Check for Drift Flag
    simulate_drift = False
    # We look for the word "drift" in the command
    if len(sys.argv) > 3 and sys.argv[3] == "drift":
        simulate_drift = True

    # 2. LOAD DATA
    data = load_wine(as_frame=True)
    df = data.frame
    
    X = df.drop(["alcohol"], axis=1)
    y = df["alcohol"]

    # --- SIMULATE DATA DRIFT ---
    # If the user typed "drift", we ruin the data
    if simulate_drift:
        print("\n⚠️ WARNING: Simulating Data Drift! Adding noise to features...")
        # We mess up the data by adding random large numbers
        X = X + np.random.normal(0, 10, X.shape) 
    # ---------------------------

    # Split
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=40)

    print(f"Training with alpha={alpha}, l1_ratio={l1_ratio}, drift={simulate_drift}...")
    
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"  RMSE: {rmse}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(lr, "model")

        # --- QUALITY GATE (The MLOps Guardrail) ---
        # If the model is bad (RMSE > 0.6), we fail the pipeline
        threshold = 0.6
        if rmse > threshold:
            print(f"\n❌ FAILURE: Model RMSE ({rmse:.3f}) is higher than threshold ({threshold})!")
            print("Blocking deployment...")
            # This line causes the crash we want
            raise Exception("Model Quality Gate Failed")
        else:
            print("\n✅ SUCCESS: Model passed quality gate.")