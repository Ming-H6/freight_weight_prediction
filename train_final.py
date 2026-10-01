# 1.Imports
import pandas as pd
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

# 5.Prepare full training data

X = X.copy()
weight_median = X["weight"].median()
X["weight"] = X["weight"].fillna(weight_median)
market_index_median = X["market_index"].median()
X["market_index"] = X["market_index"].fillna(market_index_median)
quote_signal_median = X["quote_signal"].median()

# 6.Encode equipment
encoder = OneHotEncoder(handle_unknown="ignore")
X_equipment = encoder.fit_transform(X[["equipment"]])
X_sparse_matrix = csr_matrix(X[["distance","weight","market_index","quote_signal","month","day_of_week"]].values)
X_final = hstack([X_equipment,X_sparse_matrix])


# 7.Build final fueature matrix
model = XGBRegressor(
    n_estimators = 300,
    max_depth = 5,
    learning_rate = 0.05,
    random_state = 42
)

# 8.Train final model
y_log = np.log1p(y)
model = model.fit(X_final,y_log)

# 9.load validation data
validation_df = pd.read_csv("data/validation.csv")
december_df = pd.read_csv('data/december-chart-inputs.csv')

validation_df["weight"] = validation_df["weight"].abs()
validation_df["date"] = pd.to_datetime(validation_df["date"])
validation_df["month"] = validation_df["date"].dt.month
validation_df["day_of_week"] = validation_df["date"].dt.dayofweek
validation_df["weight"] = validation_df["weight"].fillna(weight_median)
validation_df["market_index"] = validation_df["market_index"].fillna(market_index_median)

december_df["date"] = pd.to_datetime(december_df["date"])
december_df["month"] = december_df["date"].dt.month
december_df["day_of_week"] = december_df["date"].dt.dayofweek
december_df["market_index"] = market_index_median
december_df["quote_signal"] = quote_signal_median

# 10.Encode validation equipment
validation_equipment = encoder.transform(validation_df[["equipment"]])
validation_sparse_matrix = csr_matrix(validation_df[["distance","weight","market_index","quote_signal","month","day_of_week"]].values)
validation_final = hstack([validation_equipment,validation_sparse_matrix])


december_equipment = encoder.transform(december_df[["equipment"]])
december_sparse_matrix = csr_matrix(december_df[["distance","weight","market_index","quote_signal","month","day_of_week"]].values)
december_final = hstack([december_equipment,december_sparse_matrix])

# 11.Predict validation data
validation_pred_log = model.predict(validation_final)
validation_pred = np.expm1(validation_pred_log)

december_pred_log = model.predict(december_final)
december_pred = np.expm1(december_pred_log)
# 12. Output prediction file
output = pd.DataFrame({
    "load_id":validation_df["load_id"],
    "predicted_rate":validation_pred
})

output.to_csv('validation_predictions.csv',index=False)

output_december = pd.DataFrame({
    "pickup":december_df["pickup"],
    "delivery":december_df["delivery"],
    "distance":december_df["distance"],
    "equipment":december_df["equipment"],
    "weight":december_df["weight"],
    "date":december_df["date"],
    "predicted_rate":december_pred
})

output_december.to_csv('december-chart-outputs.csv',index = False)


