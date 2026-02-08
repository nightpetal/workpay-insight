import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

drop_cols = [
    "Freelancer_ID",
    "Payment_Method",
]

y_cols = ["Hourly_Rate", "Job_Success_Rate", "Client_Rating"]

data = pd.read_csv("freelancer_earnings_bd.csv")
data.drop(columns=drop_cols, inplace=True)

data["Experience_Level"] = pd.Categorical(
    data["Experience_Level"],
    categories=["Beginner", "Intermediate", "Expert"],
    ordered=True,
)
le = LabelEncoder()
data["Experience_Level_encoded"] = le.fit_transform(data["Experience_Level"])
data.drop(columns=["Experience_Level"], inplace=True)

data["Project_Type"] = pd.Categorical(
    data["Project_Type"],
    categories=["Hourly", "Fixed"],
    ordered=True,
)
le = LabelEncoder()
data["Project_Typeencoded"] = le.fit_transform(data["Project_Type"])
data.drop(columns=["Project_Type"], inplace=True)

# -----------------------------
# Train / test split
# -----------------------------
X = data.drop(columns=y_cols)
y = data[y_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=20
)

nominal_cols = ["Job_Category", "Platform", "Client_Region"]

X_train = pd.get_dummies(X_train, columns=nominal_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=nominal_cols, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

models = {}
predictions = pd.DataFrame(index=X_test.index)
mae_scores = {}

for target in y_cols:
    rf = RandomForestRegressor(n_estimators=100, random_state=20)
    rf.fit(X_train, y_train[target])

    models[target] = rf
    preds = rf.predict(X_test)

    predictions[target] = preds
    mae_scores[target] = mean_absolute_error(y_test[target], preds)

print("Predictions (first 5 rows):\n", predictions.head())

for target, mae in mae_scores.items():
    print(f"MAE for {target}: {mae:.4f}")

joblib.dump({"models": models, "columns": X_train.columns}, "model.pkl")
