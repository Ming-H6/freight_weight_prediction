# Freight Rate Prediction Challenge

Machine learning solution for the Spotter Freight Rate Prediction 

## Project Overview

The goal of this project is to predict freight rates for unseen loads using historical shipment data.

The final solution uses **XGBoost regression trained on a log-transformed target**, with numerical, equipment, and date-based features.

## Model Approach

### Data preprocessing

* Converted `weight` to absolute values to handle negative values in the source data.
* Filled missing `weight` values using the training-set median.
* Filled missing `market_index` values using the training-set median.
* Extracted:

  * `month`
  * `day_of_week`
* One-hot encoded `equipment`.
* Used an 80/20 train-validation split with `random_state=42` during model selection.

### Features

The final model uses:

* `distance`
* `weight`
* `market_index`
* `quote_signal`
* `equipment`
* `month`
* `day_of_week`

The target variable `posted_rate` was transformed using:

```python
np.log1p(posted_rate)
```

Predictions were converted back to the original scale using:

```python
np.expm1(prediction)
```

### Final model

```text
XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
```

## Model Selection

Several models and feature combinations were evaluated using the same 80/20 validation split.

| Model                                   |        MAE |       RMSE |         R² |
| --------------------------------------- | ---------: | ---------: | ---------: |
| Linear Regression — distance            |     195.17 |     559.52 |     0.8536 |
| Linear Regression — numeric             |     190.86 |     556.15 |     0.8554 |
| Linear Regression — numeric + route     |     186.91 |     553.51 |     0.8567 |
| XGBoost — numeric                       |     169.58 |     579.95 |     0.8427 |
| XGBoost — log target                    |     143.29 |     547.56 |     0.8598 |
| XGBoost — log target + equipment        |     105.73 |     530.65 |     0.8683 |
| XGBoost — log target + equipment + date | **101.34** | **529.45** | **0.8689** |

The final model was selected based on the validation results above.

## Validation

The final model was retrained on the complete `train-test.csv` dataset before generating the assessment predictions.

The resulting validation prediction file contains:

* 12,000 predictions
* Unique `load_id` values
* `load_id`
* `predicted_rate`

## December Scenario

The December scenario contains 31 fixed inputs.

The December input does not contain `market_index` or `quote_signal`, so the training-set medians were used for these features to keep the prediction feature set consistent with the final model.

The generated predictions are stored in:

```text
december-chart-outputs.csv
```

The scorer-generated visualization is:

```text
scorer_results/candidate_december.png
```

## Project Structure

```text
.
├── data/
│   ├── train-test.csv
│   ├── validation.csv
│   ├── december-chart-inputs.csv
│   └── validation-predictions-template.csv
│
├── baseline.py
├── train_final.py
├── validation_predictions.csv
├── december-chart-outputs.csv
├── requirements.txt
├── score.py
├── freight-rate-ml-assessment.pdf
├── freight-rate-ml-assessment.docx
├── readme.md
│
└── scorer_results/
    └── candidate_december.png
```

## How to Run

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the final model and generate both prediction files:

```bash
python train_final.py
```

Run the assessment scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-outputs.csv
```

A successful scorer run should validate:

```text
12,000 final predictions
31 fixed December predictions
```

and create:

```text
scorer_results/candidate_december.png
```

## Files

| File                                    | Description                                   |
| --------------------------------------- | --------------------------------------------- |
| `train_final.py`                        | Final training and prediction pipeline        |
| `baseline.py`                           | Model experiments and validation results      |
| `validation_predictions.csv`            | Predictions for the 12,000 validation loads   |
| `december-chart-outputs.csv`            | Predictions for the 31 December scenario rows |
| `score.py`                              | Assessment validation script                  |
| `requirements.txt`                      | Python dependencies                           |
| `scorer_results/candidate_december.png` | December prediction chart                     |

## Result

The final pipeline successfully generated and passed the required prediction-format checks:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
```

Final validation metrics reported during model selection:

```text
MAE:  101.34
RMSE: 529.45
R²:   0.8689
```
