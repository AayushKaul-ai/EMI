# -*- coding: utf-8 -*-
# Fixed app.py — safe loader with dependency checker and corrected imports

import importlib.util
import streamlit as st

# diagnostic import-checker — ensure required packages are present
need = ["joblib", "requests", "pandas", "sklearn"]
missing = [p for p in need if importlib.util.find_spec(p) is None]

if missing:
    st.set_page_config(page_title="Dependency error")
    st.title("Missing Python packages")
    st.error("Required packages not installed in the runtime: " + ", ".join(missing))
    st.write("Please add them to requirements.txt and push to GitHub, then redeploy.")
    st.stop()

# safe imports (now that we've checked availability)
import joblib, io, requests, os
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="EMI App (safe load)", layout="centered")
st.title("EMI App — safe loader")

st.write("App loaded. Click the button below to load model files (so page won't be blank).")

# ---------- helper to get secrets safely ----------
def get_secret(key):
    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key)

# ---------- cached loader (runs once) ----------
@st.cache_resource
def load_model_and_preprocessor(local_bundle="best_bundle_classification.pkl",
                                local_feat="feature_names_classification.pkl",
                                remote_bundle_url=None,
                                remote_feat_url=None):
    """Return (bundle, feature_names) or raise exception."""
    # try local
    if Path(local_bundle).exists() and Path(local_feat).exists():
        bundle = joblib.load(local_bundle)
        feats = joblib.load(local_feat)
        return bundle, feats

    # fallback to remote (from secrets or provided)
    remote_bundle_url = remote_bundle_url or get_secret("CLASSIFIER_BUNDLE_URL")
    remote_feat_url = remote_feat_url or get_secret("CLASSIFIER_FEAT_URL")

    if not (remote_bundle_url and remote_feat_url):
        raise FileNotFoundError("No local files and no remote URLs in secrets. Add them or upload pickles to repo.")

    # download with timeout and clear error messages
    rb = requests.get(remote_bundle_url, timeout=60)
    rb.raise_for_status()
    bundle = joblib.load(io.BytesIO(rb.content))

    rf = requests.get(remote_feat_url, timeout=60)
    rf.raise_for_status()
    feats = joblib.load(io.BytesIO(rf.content))

    return bundle, feats

# ---------- UI control: load on demand ----------
model_loaded = False
bundle = None
feature_names = None

if st.button("Load model and run quick test"):
    with st.spinner("Loading model — this may take a few seconds..."):
        try:
            # choose classifier or regressor URLs/names depending on UI state
            bundle, feature_names = load_model_and_preprocessor(
                local_bundle="best_bundle_classification.pkl",
                local_feat="feature_names_classification.pkl"
            )
            st.success("Model loaded successfully!")
            model_loaded = True
        except Exception as e:
            st.error(f"Model load failed: {e}")

# If model_loaded do rest of app (uploader, predict)
if model_loaded and bundle is not None and feature_names is not None:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            # try multiple encodings
            raw = uploaded.read()
            for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except Exception:
                    df = None
            if df is None:
                st.error("Could not parse uploaded CSV. Save as UTF-8 CSV and try again.")
                st.stop()

        st.write("Uploaded rows:", df.shape[0])
        # normalize columns (simple cleaning)
        df.columns = [str(c).strip() for c in df.columns]
        # Align features - if feature_names is a list, use it; if pkl contains list else try keys
        if isinstance(feature_names, (list, tuple)):
            required = feature_names
        elif isinstance(feature_names, dict):
            required = list(feature_names.keys())
        else:
            try:
                required = list(feature_names)
            except Exception:
                required = df.columns.tolist()

        X = df.reindex(columns=required, fill_value=0)
        try:
            preprocessor = bundle.get("preprocessor") if isinstance(bundle, dict) else None
            model = bundle.get("model") if isinstance(bundle, dict) else bundle
            if preprocessor is not None:
                X_proc = preprocessor.transform(X)
            else:
                X_proc = X.values
            preds = model.predict(X_proc)
            out = df.copy()
            out["prediction"] = preds
            st.dataframe(out.head(50))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Expected features (sample):", required[:20])
            st.write("Provided columns (sample):", list(X.columns)[:20])
else:
    st.info("Click 'Load model and run quick test' to load model and enable predictions.")