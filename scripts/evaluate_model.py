"""Recalculate the saved model's held-out test metrics (does not retrain)."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'housing.csv'
MODEL = ROOT / 'boston_rf_model.joblib'


def main():
    df = pd.read_csv(DATA)
    X, y = df[['RM','LSTAT','PTRATIO']], df['MEDV']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # Only load trusted model artifacts: joblib may execute code during deserialization.
    model = joblib.load(MODEL)
    pred = model.predict(X_test)
    print(f"Test samples: {len(X_test)}")
    print(f"Test R²: {r2_score(y_test, pred):.6f}")
    print(f"MAE: ${mean_absolute_error(y_test, pred):,.2f}")
    print(f"RMSE: ${root_mean_squared_error(y_test, pred):,.2f}")
    print("Note: reported held-out metrics are a single split, not external validation.")


if __name__ == '__main__':
    main()
