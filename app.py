import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration (This must be the first Streamlit command)
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS for a modern, polished look
st.markdown("""
    <style>
    /* Style the main button */
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
    /* Add a subtle background to the main area */
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load the trained pipeline
@st.cache_resource # This caches the model so it doesn't reload on every click
def load_model():
    try:
        with open('stroke_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# 4. Sidebar for Inputs (Keeps the main screen clean)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100) # Optional medical icon
st.sidebar.title("Patient Vitals")
st.sidebar.write("Enter the parameters below:")

# Grouping inputs logically in the sidebar
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

# 5. Main Dashboard Area
st.title("🩺 Stroke Risk Prediction Dashboard")
st.write("This tool uses machine learning to estimate the probability of a stroke based on patient demographics and health vitals.")
st.divider()

if model is None:
    st.error("⚠️ Error: 'stroke_pipeline.pkl' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# 6. Prediction Logic & Beautiful Results Display
if st.sidebar.button("Generate Diagnosis"):
    
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

    # THE BYPASS FIX: Duplicate the row to prevent the 1D/2D flattening bug
    input_data = pd.concat([input_data, input_data], ignore_index=True)

    # Run Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Display Results using Streamlit Columns and Metrics
    st.subheader("Diagnosis Results")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction[0] == 1:
            st.error("### ⚠️ High Risk Detected")
            st.metric(label="Calculated Stroke Probability", value=f"{probability:.1%}")
            st.progress(probability)
            st.warning("Recommendation: Immediate consultation with a healthcare professional is advised for a comprehensive clinical evaluation.")
        else:
            st.success("### ✅ Low Risk Detected")
            st.metric(label="Calculated Stroke Probability", value=f"{probability:.1%}")
            st.progress(probability)
            st.info("Recommendation: Maintain current healthy lifestyle habits to keep vitals in check.")
            
else:
    # Default state before clicking the button
    st.info("👈 Please enter the patient details in the sidebar and click **Generate Diagnosis** to see the results.")
