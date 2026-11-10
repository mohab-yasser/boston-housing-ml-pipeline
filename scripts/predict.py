"""Simple command-line inference using the committed educational model artifact."""
import argparse
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "boston_rf_model.joblib"
FEATURES = ("RM", "LSTAT", "PTRATIO")


def main():
    parser = argparse.ArgumentParser(description="Boston Housing educational prediction CLI")
    parser.add_argument("--rm", type=float, required=True, help="Average room count")
    parser.add_argument("--lstat", type=float, required=True, help="LSTAT (percent)")
    parser.add_argument("--ptratio", type=float, required=True, help="Pupil-teacher ratio")
    args = parser.parse_args()
    x = pd.DataFrame([[args.rm, args.lstat, args.ptratio]], columns=FEATURES)
    # Only load trusted, locally tracked model artifacts. joblib uses pickle internally.
    model = joblib.load(MODEL)
    estimate = float(model.predict(x)[0])
    print(f"Historical dataset-based estimate: ${estimate:,.2f}")
    print("Educational demo only; not a current property valuation.")


if __name__ == "__main__":
    main()
