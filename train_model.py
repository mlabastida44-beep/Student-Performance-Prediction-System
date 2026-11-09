# train_model.py
import io
import os
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------
# Part 1: Retrieve the dataset (Activity A)
# ---------------------------------------------------------------
UCI_ZIP = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def load_math_records():
    """Return the Mathematics records (student-mat.csv, 395 rows)."""
    try:
        raw = urllib.request.urlopen(UCI_ZIP, timeout=60).read()
        outer = zipfile.ZipFile(io.BytesIO(raw))
        # The UCI zip may contain a nested student.zip
        if "student.zip" in outer.namelist():
            outer = zipfile.ZipFile(io.BytesIO(outer.read("student.zip")))
        with outer.open("student-mat.csv") as f:
            return pd.read_csv(f, sep=";")
    except Exception as err:
        print("Could not read student-mat.csv from the UCI zip:", err)
        print("Falling back to ucimlrepo (fetch_ucirepo(id=320)).")
        from ucimlrepo import fetch_ucirepo

        student = fetch_ucirepo(id=320)
        return pd.concat([student.data.features, student.data.targets], axis=1)


df = load_math_records()
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print("(Mathematics file should have 395 rows; Portuguese file has 649.)")
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
assert {"G1", "G2", "G3"}.issubset(df.columns), "G1, G2 and G3 must be present"

# ---------------------------------------------------------------
# Part 2: Preprocess, train and evaluate (Activity B)
# ---------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor
X = df.drop(columns=["G3"])
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
report_text = classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
)
print(report_text)

# Summary for the "At risk" class (the one the system is meant to catch)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, predictions, labels=[1], zero_division=0
)
summary = (
    f"Accuracy:  {accuracy_score(y_test, predictions):.3f}\n"
    f"Precision (At risk): {prec[0]:.3f}\n"
    f"Recall    (At risk): {rec[0]:.3f}\n"
    f"F1-score  (At risk): {f1[0]:.3f}\n"
)
print(summary)

# Save the evaluation so it can be copied into the submission
with open("evaluation_results.txt", "w") as f:
    f.write(summary + "\n" + report_text)

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
