from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Expresso Churn Predictor", page_icon="📉", layout="wide")


def resolve_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "MODEL" / "churn_model_bundle.joblib"


@st.cache_resource
def load_bundle(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            "Model artifact not found. Train the model first with: "
            "python SCRIPTS/training_code.py --input DATA/Expresso_churn_dataset.csv"
        )

    bundle = joblib.load(model_path)
    if "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError("Invalid model bundle format. Re-train using SCRIPTS/training_code.py.")
    return bundle


def build_input_form(metadata):
    inputs = {}
    feature_columns = metadata["feature_columns"]
    categorical_features = set(metadata["categorical_features"])
    categorical_values = metadata["categorical_values"]
    numeric_defaults = metadata["numeric_defaults"]

    col1, col2 = st.columns(2)
    split_index = (len(feature_columns) + 1) // 2

    for i, feature in enumerate(feature_columns):
        container = col1 if i < split_index else col2
        with container:
            if feature in categorical_features:
                options = categorical_values.get(feature, [])
                if not options:
                    options = ["UNKNOWN"]
                inputs[feature] = st.selectbox(feature, options=options, index=0)
            else:
                inputs[feature] = st.number_input(
                    feature,
                    value=float(numeric_defaults.get(feature, 0.0)),
                    format="%.6f",
                )

    return inputs


def predict_churn(pipeline, payload: dict) -> dict:
    input_df = pd.DataFrame([payload])
    prediction = int(pipeline.predict(input_df)[0])
    churn_probability = float(pipeline.predict_proba(input_df)[0][1])
    return {
        "prediction": prediction,
        "churn_probability": churn_probability,
    }


st.title("Expresso Customer Churn Prediction")
st.caption("Portfolio-ready inference app powered by a local trained model artifact.")

model_path = resolve_model_path()

try:
    bundle = load_bundle(model_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

pipeline = bundle["pipeline"]
metadata = bundle["metadata"]

st.subheader("Enter customer features")
input_data = build_input_form(metadata)

if st.button("Predict", use_container_width=True):
    try:
        result = predict_churn(pipeline, input_data)
        label = "CHURN" if result["prediction"] == 1 else "NO CHURN"
        st.success(f"Prediction: {label}")
        st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
