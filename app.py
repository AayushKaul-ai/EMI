# app.py — minimal smoke-test
import streamlit as st
st.set_page_config(page_title="Smoke test")
st.title("Streamlit smoke test")
st.write("Hello — Streamlit is working. If you see this, Cloud is fine.")
st.write("If this loads but your real app is blank, the issue is in app.py (model loading/long ops).")
