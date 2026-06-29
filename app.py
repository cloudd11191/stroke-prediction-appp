import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        color: white;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model (Detective Version)
@st.cache_resource
def load_model():
    try:
        with open('stroke_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        # If the model fails to load, this prints the exact missing library/error to your screen
        st.error(f"DETECTIVE ERROR: Could not load model. {e}")
        return None

model = load_model()

# 4. Sidebar Inputs
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.title("Patient Vitals")

with st.sidebar.expander("👤 Demographic Details", expanded=True):
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

with st.sidebar.expander("❤️ Health Metrics", expanded=True):
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# 5. Dashboard
st.title("🩺 Stroke Risk Prediction Dashboard")
st.write("This tool uses machine learning to estimate the probability of a stroke.")
st.divider()

if model is not None:
    if st.sidebar.button("Generate Diagnosis"):
        
        # Prepare Data
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [Residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        })

        # Double Row Bypass (Fixes the flattening bug)
        input_data = pd.concat([input_data, input_data], ignore_index=True)

        # Predict
        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            st.subheader("Diagnosis Results")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction[0] == 1:
                    st.error("### ⚠️ High Risk Detected")
                    st.metric(label="Calculated Stroke Probability", value=f"{probability:.1%}")
                    st.progress(probability)
                else:
                    st.success("### ✅ Low Risk Detected")
                    st.metric(label="Calculated Stroke Probability", value=f"{probability:.1%}")
                    st.progress(probability)
        except Exception as e:
            st.error(f"PREDICTION ERROR: {e}")
            
else:
    st.info("👈 Please enter the patient details in the sidebar and click **Generate Diagnosis**.")
