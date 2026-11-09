# app.py  -- Student Performance Prediction (Streamlit)
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")

# Always look for files next to app.py, no matter where Streamlit is launched from
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_model.pkl"
DATA_PATH = BASE_DIR / "student_data.csv"

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration")

# --- Friendly checks instead of a crash --------------------------------
if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.warning(
        "`student_model.pkl` or `student_data.csv` was not found. "
        "Click the button to run `train_model.py` (needs internet to download the dataset)."
    )
    if st.button("Train model now"):
        script = BASE_DIR / "train_model.py"
        if not script.exists():
            st.error("train_model.py must be in the same folder as app.py.")
            st.stop()
        with st.spinner("Downloading data and training the model..."):
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            st.error("Training failed. See the details below.")
            st.code(result.stdout[-2000:] + "\n" + result.stderr[-3000:])
            st.stop()
        st.success("Training finished.")
        st.code(result.stdout[-1500:])
        st.rerun()
    st.stop()


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)


try:
    model = load_model(str(MODEL_PATH))
except Exception as err:
    st.error(
        "The saved model could not be loaded (often a scikit-learn version "
        "mismatch). Re-run `python train_model.py` with the same Python "
        f"environment you use for Streamlit.\n\nDetails: {err}"
    )
    st.stop()

data = load_data(str(DATA_PATH))

st.info(
    "Predictions are not diagnoses and must not be used as the sole basis "
    "for real student decisions."
)

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(float(data["G3"].mean()), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10))

st.subheader("Predict a single student")
st.write("Enter one student's details to estimate academic risk.")

feature_cols = list(model.feature_names_in_)
feature_data = data[feature_cols]

with st.form("single_student_form"):
    form_cols = st.columns(3)
    values = {}
    for i, col in enumerate(feature_cols):
        target = form_cols[i % 3]
        series = feature_data[col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            lo, hi, med = series.min(), series.max(), series.median()
            if pd.api.types.is_integer_dtype(series):
                values[col] = target.number_input(
                    col, min_value=int(lo), max_value=int(hi),
                    value=int(round(med)), step=1,
                )
            else:
                values[col] = target.number_input(
                    col, min_value=float(lo), max_value=float(hi),
                    value=float(med),
                )
        else:
            options = sorted(series.astype(str).unique().tolist())
            default = series.astype(str).mode().iloc[0]
            values[col] = target.selectbox(
                col, options, index=options.index(default)
            )
    submitted = st.form_submit_button("Predict risk")

if submitted:
    single = pd.DataFrame([values])[feature_cols]
    try:
        label = int(model.predict(single)[0])
        risk_prob = float(model.predict_proba(single)[0][1])
    except Exception as err:
        st.error(f"Prediction failed. Details: {err}")
    else:
        if label == 1:
            st.error("Prediction: AT RISK")
        else:
            st.success("Prediction: NOT AT RISK")
        st.metric("Estimated probability of being at risk", f"{risk_prob:.0%}")
        st.progress(min(max(risk_prob, 0.0), 1.0))
        st.caption(
            "This is a statistical estimate from historical data, not a diagnosis. "
            "Use it only as a prompt to review the student's situation."
        )

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")

template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    file_name="sample_prediction_input.csv",
    mime="text/csv",
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    try:
        # sep=None auto-detects comma or semicolon (the original UCI file uses ';')
        new_data = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception as err:
        st.error(f"Could not read the uploaded file as CSV: {err}")
        st.stop()

    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        try:
            predictions = model.predict(new_data[required_columns])
        except Exception as err:
            st.error(f"Prediction failed. Check the column values. Details: {err}")
            st.stop()

        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK" for value in predictions
        ]

        st.subheader("Prediction results")
        st.dataframe(results)
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            file_name="student_predictions.csv",
            mime="text/csv",
        )
