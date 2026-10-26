import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Mobile Money Fraud Detection System",
    page_icon="💰",
    layout="wide"
)

FEATURES = [
    "amount",
    "oldbalance_sender",
    "newbalance_sender",
    "oldbalance_receiver",
    "newbalance_receiver",
]

# ---------------------------------------------------------------
# Synthetic data generation (with realistic fraud / legit patterns)
# ---------------------------------------------------------------
def make_legit(n, rng):
    old_s = rng.integers(20_000, 500_000, n)
    amount = (old_s * rng.uniform(0.02, 0.40, n)).astype(int) + 500
    amount = np.minimum(amount, old_s)
    new_s = old_s - amount
    old_r = rng.integers(0, 300_000, n)
    new_r = old_r + amount
    return pd.DataFrame({
        "amount": amount,
        "oldbalance_sender": old_s,
        "newbalance_sender": new_s,
        "oldbalance_receiver": old_r,
        "newbalance_receiver": new_r,
        "actual_label": 0,
    })


def make_fraud(n, rng):
    # Typical fraud: sender account drained, funds sent to an empty account
    old_s = rng.integers(50_000, 500_000, n)
    amount = (old_s * rng.uniform(0.90, 1.00, n)).astype(int)
    new_s = old_s - amount
    old_r = rng.integers(0, 5_000, n)
    new_r = np.where(rng.random(n) < 0.5, old_r + amount, 0)  # often cashed out
    return pd.DataFrame({
        "amount": amount,
        "oldbalance_sender": old_s,
        "newbalance_sender": new_s,
        "oldbalance_receiver": old_r,
        "newbalance_receiver": new_r,
        "actual_label": 1,
    })


def generate_synthetic_data(scenario, n=12, seed=42):
    rng = np.random.default_rng(seed)
    if scenario == "✅ All Legitimate":
        df = make_legit(n, rng)
    elif scenario == "🚨 All Fraudulent":
        df = make_fraud(n, rng)
    elif scenario == "⚖️ Mixed (50% fraud)":
        n_f = n // 2
        df = pd.concat([make_legit(n - n_f, rng), make_fraud(n_f, rng)])
    else:  # "🔀 Mixed (mostly legitimate, ~20% fraud)"
        n_f = max(1, int(n * 0.2))
        df = pd.concat([make_legit(n - n_f, rng), make_fraud(n_f, rng)])

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{str(i + 1).zfill(3)}" for i in range(len(df))])
    return df


# ---------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("random_forest_model.joblib"), True
    except Exception:
        # Demonstration model trained on the same synthetic patterns
        rng = np.random.default_rng(2024)
        train = pd.concat([make_legit(2000, rng), make_fraud(2000, rng)])
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(train[FEATURES], train["actual_label"])
        return model, False


model, is_real_model = load_model()

# ---------------------------------------------------------------
# Session state (keeps data across Streamlit reruns)
# ---------------------------------------------------------------
st.session_state.setdefault("data", None)
st.session_state.setdefault("results", None)
st.session_state.setdefault("seed", 42)

# ---------------------------------------------------------------
# Header & sidebar
# ---------------------------------------------------------------
st.title("Machine Learning-Based Fraud Detection System")
st.markdown("### Nigerian Mobile Money Transactions")
st.markdown("Upload transactions and find out!")

if not is_real_model:
    st.warning("Real model not found. Using a demonstration model.")

st.sidebar.header("About the System")
st.sidebar.info(
    """
    This system uses a **Random Forest** machine learning model
    to detect fraudulent mobile money transactions.

    **Developed by:**
    Bassey Emmanuel Idongesit
    Matric No: U2022/5570206
    University of Port Harcourt
    """
)

st.divider()

# ---------------------------------------------------------------
# Data source
# ---------------------------------------------------------------
option = st.radio(
    "Choose Data Source:",
    ("Upload CSV File", "Use Synthetic Data (Recommended)"),
)

if option == "Upload CSV File":
    uploaded_file = st.file_uploader("📂 Upload Transaction CSV File", type=["csv"])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.results = None

else:
    c1, c2 = st.columns([2, 1])
    scenario = c1.selectbox(
        "Choose a scenario:",
        [
            "✅ All Legitimate",
            "🚨 All Fraudulent",
            "⚖️ Mixed (50% fraud)",
            "🔀 Mixed (mostly legitimate, ~20% fraud)",
        ],
    )
    n_rows = c2.slider("Number of transactions", 5, 100, 12)

    b1, b2 = st.columns(2)
    if b1.button("Generate Synthetic Data", type="primary"):
        st.session_state.data = generate_synthetic_data(scenario, n_rows, st.session_state.seed)
        st.session_state.results = None
        st.success("Synthetic data generated successfully!")
    if b2.button("🎲 Generate a New Random Batch"):
        st.session_state.seed = int(np.random.randint(0, 1_000_000))
        st.session_state.data = generate_synthetic_data(scenario, n_rows, st.session_state.seed)
        st.session_state.results = None
        st.success("New random batch generated!")

# ---------------------------------------------------------------
# Show data + detect
# ---------------------------------------------------------------
data = st.session_state.data

if data is not None:
    st.subheader("📄 Transaction Data")
    st.dataframe(data, use_container_width=True)

    if st.button("🔍 Detect Fraud", type="primary"):
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Your data is missing required columns: {', '.join(missing)}")
        else:
            with st.spinner("Analyzing transactions..."):
                features = data[FEATURES]
                predictions = model.predict(features)
                probabilities = model.predict_proba(features)[:, 1]

                results = data.copy()
                results["Fraud Probability"] = np.round(probabilities, 4)
                results["Prediction"] = predictions
                results["Status"] = results["Prediction"].apply(
                    lambda x: "🚨 Fraudulent" if x == 1 else "✅ Legitimate"
                )
                st.session_state.results = results

# Results are read from session state so they survive reruns
# (e.g. clicking the download button)
results = st.session_state.results

if results is not None:
    total = len(results)
    fraud_count = int((results["Prediction"] == 1).sum())
    legitimate_count = total - fraud_count
    fraud_percentage = round((fraud_count / total) * 100, 2)

    st.subheader("📊 Detection Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", total)
    col2.metric("Fraudulent", fraud_count)
    col3.metric("Legitimate", legitimate_count)
    col4.metric("Fraud Rate", f"{fraud_percentage}%")

    # If synthetic data (has known labels), show how well the model did
    if "actual_label" in results.columns:
        correct = int((results["actual_label"] == results["Prediction"]).sum())
        st.caption(f"Model matched the known synthetic label on {correct} of {total} transactions.")

    st.divider()

    st.subheader("📋 Complete Transaction Results")
    if fraud_count > 0:
        st.error(f"⚠️ {fraud_count} potentially fraudulent transaction(s) detected out of {total}!")
    else:
        st.success("✅ No fraudulent transactions detected.")

    def highlight_row(row):
        color = "background-color: rgba(255, 75, 75, 0.25)" if row["Prediction"] == 1 \
            else "background-color: rgba(40, 180, 99, 0.18)"
        return [color] * len(row)

    st.dataframe(results.style.apply(highlight_row, axis=1), use_container_width=True)
    st.caption("🔴 Red rows = fraudulent  |  🟢 Green rows = legitimate")

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
    )

elif data is None:
    st.info("👆 Choose a data source above to begin.")

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Machine Learning-Based Fraud Detection System for Nigerian Mobile Money <br>
        Final Year Project — University of Port Harcourt
    </div>
    """,
    unsafe_allow_html=True,
)
