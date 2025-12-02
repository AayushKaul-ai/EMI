#diagnostic import-checker — paste at the very top of app.py
import importlib.util
import streamlit as st

need = ["joblib", "requests", "pandas", "sklearn"]
missing = [p for p in need if importlib.util.find_spec(p) is None]

if missing:
    st.set_page_config(page_title="Dependency error")
    st.title("Missing Python packages")
    st.error("Required packages not installed in the runtime: " + ", ".join(missing))
    st.write("Please add them to requirements.txt and push to GitHub, then redeploy.")
    st.stop()

# If no missing packages, continue with normal imports below
import joblib, io, requests, os

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
    rb = requests.get(remote_bundle_url, timeout=30)
    rb.raise_for_status()
    bundle = joblib.load(io.BytesIO(rb.content))

    rf = requests.get(remote_feat_url, timeout=30)
    rf.raise_for_status()
    feats = joblib.load(io.BytesIO(rf.content))

    return bundle, feats

# ---------- UI control: load on demand ----------
model_loaded = False
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
if model_loaded:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)  # use safe_read_csv if you need multi-encoding
        # ... normalize columns, preproc, predict ...