import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors

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

# Cache SHAP explainer (expensive operation) separately from SHAP values
@st.cache_resource
def get_shap_explainer(_model):
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception as e:
        return None

model = load_model(model_path)

if model is None:
    st.warning("Please provide a valid model path or train the model first using main.py")
    st.stop()

# Feature input section
st.header("Patient Information")

# Get model features to determine which inputs to show
model_features = model.feature_names or []

# Create dynamic input fields based on model features
input_data = {}
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Vitals")
    if 'age' in model_features or 'AGE' in model_features:
        age = st.slider("Age", 18, 100, 50)
        input_data['age' if 'age' in model_features else 'AGE'] = age
    
    if 'male' in model_features or 'GENDER' in model_features:
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_num = 1 if gender == "Male" else 0
        input_data['male' if 'male' in model_features else 'GENDER'] = gender_num
    
    if 'sysBP' in model_features or 'AP_HIGH' in model_features:
        ap_high = st.slider("Systolic Blood Pressure", 90, 200, 120)
        input_data['sysBP' if 'sysBP' in model_features else 'AP_HIGH'] = ap_high
    
    if 'diaBP' in model_features or 'AP_LOW' in model_features:
        ap_low = st.slider("Diastolic Blood Pressure", 60, 140, 80)
        input_data['diaBP' if 'diaBP' in model_features else 'AP_LOW'] = ap_low
    
    if 'BMI' in model_features:
        bmi = st.slider("BMI", 15, 50, 25)
        input_data['BMI'] = bmi
    
    if 'heartRate' in model_features:
        heart_rate = st.slider("Heart Rate", 40, 200, 75)
        input_data['heartRate'] = heart_rate

with col2:
    st.subheader("Lab Results & Lifestyle")
    if 'totChol' in model_features or 'CHOLESTEROL' in model_features:
        cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
        cholesterol_num = 1 if cholesterol == "Normal" else (2 if cholesterol == "Above Normal" else 3)
        input_data['totChol' if 'totChol' in model_features else 'CHOLESTEROL'] = cholesterol_num
    
    if 'glucose' in model_features or 'GLUCOSE' in model_features:
        glucose = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
        glucose_num = 1 if glucose == "Normal" else (2 if glucose == "Above Normal" else 3)
        input_data['glucose' if 'glucose' in model_features else 'GLUCOSE'] = glucose_num
    
    if 'currentSmoker' in model_features or 'SMOKE' in model_features:
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        smoking_num = 1 if smoking == "Yes" else 0
        input_data['currentSmoker' if 'currentSmoker' in model_features else 'SMOKE'] = smoking_num
    
    if 'BPMeds' in model_features:
        bp_meds = st.selectbox("Blood Pressure Medication", ["No", "Yes"])
        bp_meds_num = 1 if bp_meds == "Yes" else 0
        input_data['BPMeds'] = bp_meds_num
    
    if 'prevalentStroke' in model_features:
        stroke = st.selectbox("History of Stroke", ["No", "Yes"])
        stroke_num = 1 if stroke == "Yes" else 0
        input_data['prevalentStroke'] = stroke_num
    
    if 'prevalentHyp' in model_features:
        hypertension = st.selectbox("History of Hypertension", ["No", "Yes"])
        hypertension_num = 1 if hypertension == "Yes" else 0
        input_data['prevalentHyp'] = hypertension_num
    
    if 'diabetes' in model_features:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        diabetes_num = 1 if diabetes == "Yes" else 0
        input_data['diabetes'] = diabetes_num
    
    if 'education' in model_features:
        education = st.slider("Education Level", 1, 4, 2)
        input_data['education'] = education
    
    if 'cigsPerDay' in model_features:
        cigs = st.slider("Cigarettes per Day", 0, 70, 0)
        input_data['cigsPerDay'] = cigs

# Create input DataFrame
input_data = pd.DataFrame([input_data])

# Fill missing features with default values
for feature in model_features:
    if feature not in input_data.columns:
        if feature in ['PULSE_PRESSURE', 'AGE_GROUP', 'BP_CATEGORY', 'RISK_SCORE']:
            # These are engineered features - calculate them if possible
            if feature == 'PULSE_PRESSURE' and 'sysBP' in input_data.columns and 'diaBP' in input_data.columns:
                input_data['PULSE_PRESSURE'] = input_data['sysBP'] - input_data['diaBP']
            elif feature == 'AGE_GROUP' and 'age' in input_data.columns:
                def categorize_age(age):
                    if age < 30: return 0
                    elif age < 40: return 1
                    elif age < 50: return 2
                    elif age < 60: return 3
                    elif age < 70: return 4
                    else: return 5
                input_data['AGE_GROUP'] = input_data['age'].apply(categorize_age)
            elif feature == 'BP_CATEGORY' and 'sysBP' in input_data.columns and 'diaBP' in input_data.columns:
                def categorize_bp(row):
                    systolic = row['sysBP']
                    diastolic = row['diaBP']
                    if systolic < 120 and diastolic < 80: return 0
                    elif systolic < 140 or diastolic < 90: return 1
                    elif systolic < 160 or diastolic < 100: return 2
                    else: return 3
                input_data['BP_CATEGORY'] = input_data.apply(categorize_bp, axis=1)
            elif feature == 'RISK_SCORE' and 'totChol' in input_data.columns and 'glucose' in input_data.columns and 'age' in input_data.columns:
                def calculate_risk_score(row):
                    cholesterol_score = (row['totChol'] - 1) / 2
                    glucose_score = (row['glucose'] - 1) / 2
                    age_score = row['age'] / 100
                    return cholesterol_score + glucose_score + age_score
                input_data['RISK_SCORE'] = input_data.apply(calculate_risk_score, axis=1)
            else:
                input_data[feature] = 0
        else:
            input_data[feature] = 0

# Predict button
if st.button("Predict Heart Disease Risk", type="primary"):
    model_features = model.feature_names or []
    missing_features = [f for f in model_features if f not in input_data.columns]
    if missing_features:
        st.error(f"The model requires these missing features: {missing_features}")
        st.stop()

    prediction_data = input_data[model_features].copy()

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
        # Use cached SHAP explainer to prevent repeated fitting
        explainer = get_shap_explainer(model.model)
        
        if explainer is None:
            st.warning("Could not initialize SHAP explainer")
            st.stop()
        
        # SHAP force plot
        st.subheader("Feature Contribution")
        
        # Calculate SHAP values for the prediction
        shap_values = explainer.shap_values(prediction_data)
        
        # Handle different shap_values formats (binary classification returns list, single class returns array)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Validate shap_values shape
        if len(shap_values) == 0:
            st.warning("No SHAP values calculated")
            st.stop()
        
        instance_shap = shap_values[0]
        instance_data = prediction_data.iloc[0]
        base_value = explainer.expected_value
        
        # Handle base_value for binary classification (might be array)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        
        # Use waterfall plot instead of force plot for better Streamlit rendering
        fig = plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=instance_shap,
                base_values=base_value,
                data=instance_data.values,
                feature_names=model_features
            ),
            show=False,
            max_display=10
        )
        plt.title('Feature Contribution (SHAP Waterfall Plot)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Feature importance
        st.subheader("Top Feature Contributions")
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Validate mean_shap shape
        if len(mean_shap) != len(model_features):
            st.warning("SHAP values shape mismatch with features")
            st.stop()
        
        importance_df = pd.DataFrame({
            'feature': model_features,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importance_df)), importance_df['shap_importance'].values)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'].values)
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
