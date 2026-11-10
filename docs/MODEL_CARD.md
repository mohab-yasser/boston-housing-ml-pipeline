# Model card — Boston Housing regression demo

## Intended use

Educational demonstration of the supervised regression workflow, including train/validation/test splitting, cross-validation, hyperparameter search and simple Streamlit inference. **Not intended for real estate valuations, credit, housing eligibility or other consequential decisions.**

## Model and inputs

- Artifact: `boston_rf_model.joblib`, a `RandomForestRegressor` serialized using scikit-learn **1.6.0**.
- Parameters: `n_estimators=200`, `max_depth=6`, `min_samples_leaf=1`, `min_samples_split=2`, `random_state=1`.
- Features, in order: `RM` (rooms), `LSTAT` (historical socioeconomic status feature), `PTRATIO` (pupil–teacher ratio).
- Target: `MEDV`, historical median house value in **inflation-adjusted dollar labels supplied by the modified Udacity exercise**. These are not observed 2026 transaction prices.

## Data provenance

The repository includes a 489-row, 4-column educational derivative of the legacy Boston Housing dataset. The Udacity instructional adaptation removes 16 capped target records and one unusual RM record, keeps three predictors and scales the target. The original data dates from the 1970s.

Source context: [Boston Housing educational exercise](https://notebook.community/aktse/udacity-mlnd/projects/boston_housing/boston_housing). See also [scikit-learn's deprecation warning](https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html).

## Evaluation protocol and reported results

- Initial development: single train/validation split for baseline, linear regression, decision tree and random forest.
- Model selection: 5-fold shuffled cross-validation (`KFold(n_splits=5, shuffle=True, random_state=1)`) and GridSearchCV on development data.
- Final holdout: `train_test_split(test_size=0.2, random_state=1)`; saved model scores are checked on the 98-row test portion.

| Metric | Saved-model holdout |
| --- | ---: |
| R² | 0.829353 |
| Mean absolute error | $48,555.72 |
| Root mean squared error | $68,146.87 |
| Model selection CV R² | 0.840004 |

These metrics are measured on this particular benchmark split; **there is no external validation or evidence of modern market accuracy**. Comparison-bar values from a single validation split and cross-validation represent different evaluation protocols and should not be ranked without that context.

## Ethical considerations and limitations

- The underlying Boston dataset has documented ethical problems; scikit-learn removed its bundled loader and discourages its use except for teaching data ethics.
- `LSTAT` is a socioeconomic feature and can encode historically biased patterns. Do not use it to justify individual property prices or decisions about people.
- Dataset is limited in time, geography, number of rows, and feature coverage. There is no drift monitoring, uncertainty calibration, or real-world deployment evaluation.
- Prices reflect a past educational adjustment. Labels in dollars are **not current market estimates**.
- Stored joblib files use pickle internals and must only be loaded from a trusted source. For compatibility, use scikit-learn 1.6.0 (same as the saved artifact); other versions are not supported.

## Reproducing results

From the repository root in a Python 3.11 virtual environment:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/evaluate_model.py
python -m pytest -q
jupyter lab
```

The notebook's final cell may overwrite the tracked artifact if you run all cells. A full retraining (GridSearchCV) can take materially longer than the evaluation script.
