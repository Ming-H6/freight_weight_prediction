"""
Spotter Freight Rate Assessment - Baseline Experiments

Experiments and validation results:

1. Linear Regression - distance only
   MAE: 195.17
   RMSE: 559.52
   R²: 0.8536

2. Linear Regression - numeric features
   MAE: 190.86
   RMSE: 556.15
   R²: 0.8554

3. Linear Regression - numeric + route
   MAE: 186.91
   RMSE: 553.51
   R²: 0.8567

4. XGBoost - numeric features
   MAE: 169.58
   RMSE: 579.95
   R²: 0.8427

5. XGBoost - log target
   MAE: 143.29
   RMSE: 547.56
   R²: 0.8598

6. XGBoost - log target + equipment
   MAE: 105.73
   RMSE: 530.65
   R²: 0.8683

7. XGBoost - log target + equipment + date features
   MAE: 101.34
   RMSE: 529.45
   R²: 0.8689

Final model:
XGBoost + log target + equipment + month + day_of_week
"""
# 1.Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error,r2_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import numpy as np
from scipy.sparse import csr_matrix
from xgboost import XGBRegressor



# 2.Load Data
df = pd.read_csv('data/train-test.csv')


# 3.Deterministic data preprocessing
df["weight"] = df["weight"].abs()
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

# 4.Define X and y
X = df[["distance","weight","market_index","quote_signal","equipment","month","day_of_week"]]
y = df["posted_rate"]

# 5.Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

# 6.fit preprocessing on training data
weight_median = X_train["weight"].median()
X_train["weight"] = X_train["weight"].fillna(weight_median)
X_val["weight"] = X_val["weight"].fillna(weight_median)
market_index_median = X_train["market_index"].median()
X_train["market_index"] = X_train["market_index"].fillna(market_index_median)
X_val["market_index"] = X_val["market_index"].fillna(market_index_median)
y_train_log = np.log1p(y_train)

encoder = OneHotEncoder(handle_unknown="ignore")
X_train_equipment = encoder.fit_transform(X_train[["equipment"]])
X_val_equipment = encoder.transform(X_val[["equipment"]])
X_train_sparse_matrix = csr_matrix(X_train[["distance","weight","market_index","quote_signal","month","day_of_week"]].values)
X_val_sparse_matrix = csr_matrix(X_val[["distance","weight","market_index","quote_signal","month","day_of_week"]].values)
X_train_final = hstack([X_train_equipment,X_train_sparse_matrix])
X_val_final = hstack([X_val_equipment,X_val_sparse_matrix])

# 7.Build final fueature metrics
model = XGBRegressor(
    n_estimators = 300,
    max_depth = 5,
    learning_rate = 0.05,
    random_state = 42
)

# 8.Train final model
y_train_log = np.log1p(y_train)
model = model.fit(X_train_final,y_train_log)


# 9. predict
y_pred_log = model.predict(X_val_final)
y_pred = np.expm1(y_pred_log)

# 10. Evaluate

mae = mean_absolute_error(y_val, y_pred)
rmse = root_mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val,y_pred)

print("MAE: ",mae)
print("RMSE: ",rmse)
print("R² ",r2)



