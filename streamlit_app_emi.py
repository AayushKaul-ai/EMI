 # app.py (minimal)
import streamlit as st
import joblib, os
import pandas as pd
import numpy as np
import unicodedata, re

# helper
def clean_col_name(s):
    s = str(s)
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\x00-\x1f\x7f-\x9f\uFEFF]', '', s)
    s = s.replace('“', '"').replace('”', '"').replace('’', "'").replace('–','-').replace('—','-')
    s = re.sub(r'[^\w]', '_', s)
    s = re.sub(r'__+', '_', s)
    s = s.strip('_').lower()
    return s

def normalize_cols(df):
    df = df.copy()
    df.columns = [clean_col_name(c) for c in df.columns]
    return df

@st.cache_resource
def load_bundle(bundle_path, feat_path):
    if not os.path.exists(bundle_path) or not os.path.exists(feat_path):
        raise FileNotFoundError(f"Missing {bundle_path} or {feat_path}")
    b = joblib.load(bundle_path)
    f = joblib.load(feat_path)
    return b, f

# UI
st.title("EMI Model Demo")

model_type = st.selectbox("Model type", ["classification", "regression"])

# Load bundle
if model_type == "classification":
    bundle_path = "best_bundle_classification.pkl"
    feat_path = "feature_names_classification.pkl"
else:
    bundle_path = "best_bundle_regression.pkl"
    feat_path = "feature_names_regression.pkl"

try:
    bundle, feature_names = load_bundle(bundle_path, feat_path)
except Exception as e:
    st.error(f"Model files missing or load error: {e}")
    st.stop()

preprocessor = bundle["preprocessor"]
model = bundle["model"]

st.write("Loaded model:", type(model).__name__)

# Let user upload a CSV or use a simple form
uploaded = st.file_uploader("Upload CSV with input rows (columns will be normalized)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    df = normalize_cols(df)
    X_aligned = df.reindex(columns=feature_names, fill_value=0)
    # optional type coercion:
    # numeric_cols = [c for c in feature_names if c in df.select_dtypes(include='number').columns]
    try:
        X_proc = preprocessor.transform(X_aligned)
        preds = model.predict(X_proc)
        st.write("Predictions:", preds.tolist())
    except Exception as e:
        st.error(f"Error during transform/predict: {e}")
else:
    st.info("Upload a CSV to run batch predictions, or wire up a manual form for single-row input.")