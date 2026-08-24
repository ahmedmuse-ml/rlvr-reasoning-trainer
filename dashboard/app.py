import json
import streamlit as st
from pathlib import Path
from dashboard.metrics import load_training_curves, load_eval_results, load_sample_completions

st.set_page_config(page_title="RLVR Dashboard", layout="wide")
st.title("RLVR Reasoning Trainer — Evaluation Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Before / After Accuracy")
    results = load_eval_results()
    if results:
        st.bar_chart(results)
    else:
        st.info("Weli lama helin natiijooyin eval ah. Fuli `scripts/evaluate.py` marka hore.")

with col2:
    st.subheader("Training Curves")
    curves = load_training_curves()
    if curves is not None and not curves.empty:
        available_cols = [c for c in ["reward", "kl", "loss"] if c in curves.columns]
        if available_cols:
            st.line_chart(curves[available_cols])
        else:
            st.warning(f"Columns la heli waayey. Waxaad haysataa: {list(curves.columns)}")
    else:
        st.info("Weli lama helin training logs. Fuli `scripts/train.py` marka hore.")

st.subheader("Sample Completions (Before vs After)")
samples = load_sample_completions()
if samples:
    idx = st.number_input("Sample #", min_value=0, max_value=len(samples) - 1, value=0)
    left, right = st.columns(2)
    with left:
        st.markdown("**Before Training**")
        st.text_area("", samples[idx]["before"], height=200, key="before_box")
    with right:
        st.markdown("**After Training**")
        st.text_area("", samples[idx]["after"], height=200, key="after_box")
else:
    st.info("not ready sample completions (`reports/samples.json`).")