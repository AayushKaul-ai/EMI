 # app.py (minimal)
import streamlit as st
from pathlib import Path
import joblib, io, requests, os

APP_DIR = Path(__file__).resolve().parent

st.title("EMI Model Demo")

# ---------- helpers ----------
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
def load_bundle_with_feature_names(bundle_rel_path, feat_rel_path, remote_bundle_url=None, remote_feat_url=None):
    """
    Load model bundle and feature_names. Try local first, then remote URLs if provided.
    Returns: (bundle_dict, feature_names_list)
    """
    bundle_path = APP_DIR.joinpath(bundle_rel_path)
    feat_path = APP_DIR.joinpath(feat_rel_path)

    # Try local
    if bundle_path.exists() and feat_path.exists():
        b = joblib.load(bundle_path)
        f = joblib.load(feat_path)
        return b, f

    # Try remote bundle and feature names (if URLs provided in secrets)
    if remote_bundle_url and remote_feat_url:
        # download bundle
        rb = requests.get(remote_bundle_url, timeout=30)
        rb.raise_for_status()
        b = joblib.load(io.BytesIO(rb.content))
        # download feature names
        rf = requests.get(remote_feat_url, timeout=30)
        rf.raise_for_status()
        f = joblib.load(io.BytesIO(rf.content))
        return b, f

    # Not found
    raise FileNotFoundError(f"Missing files locally: {bundle_path} or {feat_path}. Provide files or set remote URLs in st.secrets.")

# ---------- UI: choose model ----------
model_type = st.selectbox("Model type", ["classification", "regression"])

if model_type == "classification":
    bundle_rel = "best_bundle_classification.pkl"
    feat_rel = "feature_names_classification.pkl"
    remote_bundle = st.secrets.get("CLASSIFIER_BUNDLE_URL")
    remote_feat = st.secrets.get("CLASSIFIER_FEAT_URL")
else:
    bundle_rel = "best_bundle_regression.pkl"
    feat_rel = "feature_names_regression.pkl"
    remote_bundle = st.secrets.get("REGRESSOR_BUNDLE_URL")
    remote_feat = st.secrets.get("REGRESSOR_FEAT_URL")

# ---------- load model (cached) ----------
try:
    bundle, feature_names = load_bundle_with_feature_names(bundle_rel, feat_rel,
                                                           remote_bundle_url=remote_bundle,
                                                           remote_feat_url=remote_feat)
except Exception as e:
    st.error(f"Model files missing or load error: {e}")
    st.info(
        "Place the model pkl files in the app folder (same as app.py) or set remote URLs in Streamlit Secrets."
    )
    st.stop()

preprocessor = bundle["preprocessor"]
model = bundle["model"]
st.write("Loaded model:", type(model).__name__)

# ---------- file upload / inference ----------
uploaded = st.file_uploader("Upload CSV (columns will be normalized)", type=["csv"])
if uploaded:
    try:
        # Try common encodings automatically
        raw = uploaded.read()
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            st.error("Could not parse uploaded CSV. Try saving it as UTF-8 CSV and upload again.")
        else:
            df = normalize_cols(df)
            X_aligned = df.reindex(columns=feature_names, fill_value=0)
            # optional: coerce numeric types if needed
            try:
                X_proc = preprocessor.transform(X_aligned)
                preds = model.predict(X_proc)
                st.success("Predictions generated")
                st.write(preds.tolist())
            except Exception as e:
                st.error(f"Error during transform/predict: {e}")
                st.write("Expected features (sample):", feature_names[:20])
                st.write("Provided columns (sample):", X_aligned.columns.tolist()[:20])
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
else:
    st.info("Upload a CSV to run batch predictions, or add UI for single-row input.")