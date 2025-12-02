# EMI
Made 2 ML models wherein one told who are eligible for emi and other told for how much emi are they eligible.

## How to run locally
1. Create venv: `python -m venv .venv`
2. Activate: `. .\.venv\Scripts\Activate.ps1` (PowerShell)
3. Install: `pip install -r requirements.txt`
4. Create `D:\Labmentix\.streamlit\secrets.toml` or use environment variables for model URLs (local testing).
5. Run: `streamlit run app.py`

## Deploy
1. Push this repo to GitHub.
2. Configure Streamlit Cloud app and add secrets (model URLs).
3. Deploy from repo.
