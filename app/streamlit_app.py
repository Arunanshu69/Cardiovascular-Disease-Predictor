import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import XGBoostModel
from src.explainability import SHAPAnalyzer
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        color: #2a9d8f;
        font-size: 2rem;
        font-weight: bold;
    }
    .risk-medium {
        color: #e9c46a;
        font-size: 2rem;
        font-weight: bold;
    }
    .risk-high {
        color: #e63946;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">❤️ Heart Disease Prediction</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Model Configuration")

# Model path
model_path = st.sidebar.text_input("Model Path", "results/xgboost_model.pkl")

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        model = XGBoostModel()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_path)

if model is None:
    st.warning("Please provide a valid model path or train the model first using main.py")
    st.stop()

# Feature input section
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    age = st.slider("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    
    st.subheader("Vital Signs")
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    cholesterol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)

with col2:
    st.subheader("Health Metrics")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    
    st.subheader("Lifestyle")
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    physical_activity = st.slider("Physical Activity (hours/week)", 0, 20, 3)
    
    st.subheader("Other")
    bmi = st.slider("BMI", 15, 50, 25)
    glucose = st.slider("Glucose Level (mg/dl)", 50, 300, 100)

# Convert categorical to numerical
sex_num = 1 if sex == "Male" else 0
fasting_bs_num = 1 if fasting_bs == "Yes" else 0
smoking_num = 1 if smoking == "Yes" else 0

# ECG encoding
resting_ecg_num = 0 if resting_ecg == "Normal" else (1 if resting_ecg == "ST-T Wave Abnormality" else 2)

# Create input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_num],
    'trestbps': [resting_bp],
    'chol': [cholesterol],
    'thalach': [max_hr],
    'fbs': [fasting_bs_num],
    'restecg': [resting_ecg_num],
    'smoking': [smoking_num],
    'physical_activity': [physical_activity],
    'bmi': [bmi],
    'glucose': [glucose]
})

# Feature mapping (adjust based on actual model features)
feature_mapping = {
    'age': 'age',
    'sex': 'sex',
    'trestbps': 'trestbps',
    'chol': 'chol',
    'thalach': 'thalach',
    'fbs': 'fbs',
    'restecg': 'restecg',
    'smoking': 'smoking',
    'physical_activity': 'physical_activity',
    'bmi': 'bmi',
    'glucose': 'glucose'
}

# Predict button
if st.button("Predict Heart Disease Risk", type="primary"):
    # Get model features
    model_features = model.feature_names
    
    # Map input data to model features
    prediction_data = pd.DataFrame()
    for feature in model_features:
        if feature in input_data.columns:
            prediction_data[feature] = input_data[feature]
        else:
            # Use default value if feature not in input
            prediction_data[feature] = [0]
    
    # Make prediction
    prediction = model.predict(prediction_data)[0]
    probability = model.predict_proba(prediction_data)[0, 1]
    
    # Display results
    st.header("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
    
    with col2:
        st.metric("Probability", f"{probability:.2%}")
    
    with col3:
        if probability < 0.3:
            risk_class = "Low"
            risk_color = "risk-low"
        elif probability < 0.7:
            risk_class = "Medium"
            risk_color = "risk-medium"
        else:
            risk_class = "High"
            risk_color = "risk-high"
        st.markdown(f'<div class="{risk_color}">Risk Level: {risk_class}</div>', unsafe_allow_html=True)
    
    # Risk gauge
    st.header("Risk Assessment")
    
    fig, ax = plt.subplots(figsize=(10, 2))
    colors = ['#2a9d8f', '#e9c46a', '#e63946']
    ax.barh([0], [probability], color=colors[2] if probability > 0.7 else (colors[1] if probability > 0.3 else colors[0]), height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Risk Probability')
    ax.set_title('Heart Disease Risk Probability')
    
    # Add threshold lines
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.15, 0.6, 'Low Risk', ha='center', va='center', fontsize=10)
    ax.text(0.5, 0.6, 'Medium Risk', ha='center', va='center', fontsize=10)
    ax.text(0.85, 0.6, 'High Risk', ha='center', va='center', fontsize=10)
    
    st.pyplot(fig)
    plt.close()
    
    # SHAP explanation
    st.header("Explainability (SHAP)")
    
    try:
        shap_analyzer = SHAPAnalyzer(model.model)
        shap_analyzer.fit(prediction_data)
        
        # SHAP force plot
        st.subheader("Feature Contribution")
        
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(prediction_data)
        
        # Create force plot
        fig = plt.figure(figsize=(12, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            prediction_data.iloc[0],
            feature_names=model_features,
            show=False,
            matplotlib=True
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Feature importance
        st.subheader("Top Feature Contributions")
        
        importance_df = shap_analyzer.get_feature_importance_df()
        importance_df = importance_df.head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importance_df)), importance_df['shap_importance'][::-1])
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'][::-1])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Top 10 Feature Contributions')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")

# Information section
st.header("About This Model")
st.markdown("""
This heart disease prediction model uses:
- **XGBoost**: Gradient boosting algorithm for classification
- **GA-PSO Optimization**: Hybrid Genetic Algorithm and Particle Swarm Optimization for hyperparameter tuning
- **TabDDPM**: Diffusion model for synthetic data generation (if enabled during training)

**Disclaimer**: This tool is for educational purposes only and should not be used for medical diagnosis. 
Always consult a healthcare professional for medical advice.
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Diffusion-Based GA-PSO Optimized XGBoost Framework")
