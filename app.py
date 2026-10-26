import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Mobile Money Fraud Detection System",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 Machine Learning-Based Fraud Detection System")
st.markdown("### Nigerian Mobile Money Transactions")
st.markdown("You can either upload a CSV file or use the system’s built-in synthetic data.")

st.divider()

# Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model.joblib")
        return model
    except:
        st.warning("Real model not found. Using a demonstration model.")
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        return model

model = load_model()

# Sidebar
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

# Function to generate synthetic data
def generate_synthetic_data(n=10):
    np.random.seed(42)
    data = pd.DataFrame({
        "transaction_id": [f"TXN{str(i+1).zfill(3)}" for i in range(n)],
        "amount": np.random.randint(500, 400000, n),
        "oldbalance_sender": np.random.randint(1000, 500000, n),
        "newbalance_sender": np.random.randint(0, 400000, n),
        "oldbalance_receiver": np.random.randint(0, 200000, n),
        "newbalance_receiver": np.random.randint(0, 500000, n)
    })
    return data

# Option Selection
option = st.radio(
    "Choose Data Source:",
    ("Upload CSV File", "Use Synthetic Data (Recommended)")
)

data = None

if option == "Upload CSV File":
    uploaded_file = st.file_uploader("📂 Upload Transaction CSV File", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

elif option == "Use Synthetic Data (Recommended)":
    if st.button("Generate Synthetic Data", type="primary"):
        data = generate_synthetic_data(12)
        st.success("Synthetic data generated successfully!")

# Process Data if available
if data is not None:
    st.subheader("📄 Transaction Data")
    st.dataframe(data, use_container_width=True)

    if st.button("🔍 Detect Fraud", type="primary"):
        with st.spinner("Analyzing transactions..."):
            features = data.select_dtypes(include=[np.number])

            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]

            results = data.copy()
            results["Fraud Probability"] = np.round(probabilities, 4)
            results["Prediction"] = predictions
            results["Status"] = results["Prediction"].apply(
                lambda x: "🚨 Fraudulent" if x == 1 else "✅ Legitimate"
            )

            # Summary
            total = len(results)
            fraud_count = int(sum(predictions))
            legitimate_count = total - fraud_count
            fraud_percentage = round((fraud_count / total) * 100, 2)

            st.subheader("📊 Detection Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", total)
            col2.metric("Fraudulent", fraud_count)
            col3.metric("Legitimate", legitimate_count)
            col4.metric("Fraud Rate", f"{fraud_percentage}%")

            st.divider()

            st.subheader("📋 Detailed Prediction Results")
            st.dataframe(results, use_container_width=True)

            st.subheader("🚨 Fraud Alerts")
            fraud_cases = results[results["Prediction"] == 1]

            if len(fraud_cases) > 0:
                st.error(f"⚠️ {len(fraud_cases)} potentially fraudulent transaction(s) detected!")
                st.dataframe(fraud_cases, use_container_width=True)
            else:
                st.success("✅ No fraudulent transactions detected.")

            # Download button
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Choose a data source above to begin.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Machine Learning-Based Fraud Detection System for Nigerian Mobile Money <br>
        Final Year Project — University of Port Harcourt
    </div>
    """,
    unsafe_allow_html=True
)
