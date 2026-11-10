"""Smoke tests for the curated dataset and persisted model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['RM', 'LSTAT', 'PTRATIO']


@pytest.fixture(scope='module')
def dataset():
    return pd.read_csv(ROOT / 'housing.csv')


@pytest.fixture(scope='module')
def model():
    # Only load the trusted, repository-tracked model artifact.
    return joblib.load(ROOT / 'boston_rf_model.joblib')


def test_dataset_schema(dataset):
    assert dataset.shape == (489, 4)
    assert list(dataset.columns) == FEATURES + ['MEDV']
    assert not dataset.isna().any().any()
    assert not dataset.duplicated().any()


def test_model_schema_and_params(model):
    assert list(model.feature_names_in_) == FEATURES
    assert model.n_estimators == 200
    assert model.max_depth == 6
    assert model.random_state == 1


def test_holdout_metrics(dataset, model):
    _, x_test, _, y_test = train_test_split(
        dataset[FEATURES], dataset['MEDV'], test_size=0.2, random_state=1
    )
    y_pred = model.predict(x_test)
    assert r2_score(y_test, y_pred) == pytest.approx(0.82935286295, abs=1e-6)
    assert mean_absolute_error(y_test, y_pred) == pytest.approx(48555.71683, abs=0.02)
    assert root_mean_squared_error(y_test, y_pred) == pytest.approx(68146.86801, abs=0.02)


def test_example_predictions(model):
    examples = pd.DataFrame(
        {'RM':[5,4,8], 'LSTAT':[17,32,3], 'PTRATIO':[15,22,12]}, columns=FEATURES
    )
    actual = model.predict(examples)
    expected = [379490.299069, 241559.606138, 927348.374242]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.03)
