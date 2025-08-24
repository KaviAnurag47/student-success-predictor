import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")

# Load the saved sklearn Pipeline (preprocessor + model)
MODEL_PATH = "best_rf_model.pkl"  # model file is in the same folder as app.py
best_model = joblib.load(MODEL_PATH)

st.title("🎓 Student Success Predictor")
st.write("Provide student details to predict Pass (1) / Fail (0).")

# ---- UI (match training column names exactly) ----
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        NrSiblings = st.number_input("NrSiblings", min_value=0, max_value=20, value=1, step=1)
        WklyStudyHours = st.number_input("WklyStudyHours", min_value=0, max_value=60, value=10, step=1)
        TotalScore = st.number_input("TotalScore (0–300 if sum of 3 tests, else 0–100 as used)", min_value=0.0, max_value=300.0, value=180.0, step=1.0)

        Gender = st.selectbox("Gender", ["male", "female"])
        EthnicGroup = st.selectbox("EthnicGroup", ["group A", "group B", "group C", "group D", "group E"])

    with col2:
        ParentEduc = st.selectbox(
            "ParentEduc",
            ["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"]
        )
        LunchType = st.selectbox("LunchType", ["standard", "free/reduced"])
        TestPrep = st.selectbox("TestPrep", ["none", "completed"])
        ParentMaritalStatus = st.selectbox("ParentMaritalStatus", ["married", "single", "divorced"])
        PracticeSport = st.selectbox("PracticeSport", ["never", "sometimes", "regularly"])
        IsFirstChild = st.selectbox("IsFirstChild", ["yes", "no"])
        TransportMeans = st.selectbox("TransportMeans", ["school_bus", "private", "on_foot"])

    submitted = st.form_submit_button("🔮 Predict")

# ---- Prepare input EXACTLY like training ----
def make_input_df():
    return pd.DataFrame([{
        # numeric
        "NrSiblings": NrSiblings,
        "TotalScore": TotalScore,
        # categorical (exact names from training)
        "Gender": Gender,
        "EthnicGroup": EthnicGroup,
        "ParentEduc": ParentEduc,
        "LunchType": LunchType,
        "TestPrep": TestPrep,
        "ParentMaritalStatus": ParentMaritalStatus,
        "PracticeSport": PracticeSport,
        "IsFirstChild": IsFirstChild,
        "TransportMeans": TransportMeans,
        "WklyStudyHours": WklyStudyHours,
    }])

if submitted:
    try:
        input_df = make_input_df()

        # (Optional) sanity check: show columns the model expects
        preproc = best_model.named_steps["preprocessor"]
        expected_cols = list(preproc.transformers_[0][2]) + list(preproc.transformers_[1][2])

        missing = set(expected_cols) - set(input_df.columns)
        extra = set(input_df.columns) - set(expected_cols)

        if missing:
            st.error(f"Missing columns for the pipeline: {missing}")
        elif extra:
            st.warning(f"Extra columns (will be ignored by pipeline): {extra}")

        # Predict
        y_pred = best_model.predict(input_df)[0]
        proba_idx = np.where(best_model.named_steps["classifier"].classes_ == 1)[0][0]
        y_proba = best_model.predict_proba(input_df)[0][proba_idx]

        if y_pred == 1:
            st.success(f"✅ Predicted: PASS (probability {y_proba:.2f})")
        else:
            st.error(f"❌ Predicted: FAIL (probability {(1 - y_proba):.2f})")

        with st.expander("Show input sent to the model"):
            st.write(input_df)

        with st.expander("Show columns the model expects"):
            st.write(expected_cols)

    except Exception as e:
        st.exception(e)
