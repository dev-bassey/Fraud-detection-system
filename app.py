import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Mobile Money Fraud Detection System",
    page_icon="💰",
    layout="wide"
)

# --------------------------
# Title and Description
# --------------------------
st.title("💰 Machine Learning-Based Fraud Detection System")
st.markdown("### Nigerian Mobile Money Transactions")
st.markdown("Upload a CSV file containing mobile money transactions to detect fraudulent activities.")

st.divider()

# --------------------------
# Load or Create Model
# --------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model.joblib")
        return model
    except:
        # Create a dummy model if real model is not available (for demonstration)
        st.warning("Real model not found. Using a demonstration model.")
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        # Train on dummy data so the app can still run
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        return model

model = load_model()

# --------------------------
# Sidebar
# --------------------------
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

st.sidebar.markdown("---")
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown("1. Upload a CSV file")
st.sidebar.markdown("2. Click 'Detect Fraud'")
st.sidebar.markdown("3. View results and alerts")

# --------------------------
# File Uploader
# --------------------------
uploaded_file = st.file_uploader("📂 Upload Transaction CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        st.subheader("📄 Uploaded Transaction Data")
        st.dataframe(data.head(10), use_container_width=True)
        
        st.write(f"**Total Transactions Uploaded:** {len(data)}")

        if st.button("🔍 Detect Fraud", type="primary"):
            with st.spinner("Analyzing transactions... Please wait"):
                
                # --------------------------
                # Simple Feature Preparation
                # (In real case, apply same preprocessing used during training)
                # --------------------------
                try:
                    # Try to use the data as features (for real model)
                    features = data.select_dtypes(include=[np.number])
                    
                    if features.shape[1] == 0:
                        st.error("No numerical columns found in the file.")
                    else:
                        # Make prediction
                        predictions = model.predict(features)
                        probabilities = model.predict_proba(features)[:, 1]

                        # Add results to dataframe
                        results = data.copy()
                        results["Fraud Probability"] = np.round(probabilities, 4)
                        results["Prediction"] = predictions
                        results["Status"] = results["Prediction"].apply(
                            lambda x: "🚨 Fraudulent" if x == 1 else "✅ Legitimate"
                        )

                        # --------------------------
                        # Summary Metrics
                        # --------------------------
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

                        # --------------------------
                        # Results Table
                        # --------------------------
                        st.subheader("📋 Detailed Prediction Results")
                        st.dataframe(results, use_container_width=True)

                        # --------------------------
                        # Fraud Alerts
                        # --------------------------
                        st.subheader("🚨 Fraud Alerts")
                        fraud_cases = results[results["Prediction"] == 1]

                        if len(fraud_cases) > 0:
                            st.error(f"⚠️ {len(fraud_cases)} potentially fraudulent transaction(s) detected!")
                            st.dataframe(fraud_cases, use_container_width=True)
                        else:
                            st.success("✅ No fraudulent transactions detected.")

                        # --------------------------
                        # Download Results
                        # --------------------------
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Full Results as CSV",
                            data=csv,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error("An error occurred while processing the file.")
                    st.write(e)

    except Exception as e:
        st.error("Unable to read the CSV file. Please check the file format.")
        st.write(e)

else:
    st.info("👆 Please upload a CSV file to begin fraud detection.")

# --------------------------
# Footer
# --------------------------
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
