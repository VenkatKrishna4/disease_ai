import streamlit as st
import pickle

st.set_page_config(page_title="Disease Prediction", page_icon="⚕")

# Hiding Streamlit branding
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Background Image
background_image_url = "https://www.shutterstock.com/image-illustration/biomarker-discovery-diagnostic-prognostic-predictive-600nw-2197072529.jpg"

st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url({background_image_url});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content:"";
        position: absolute;
        top:0;
        left:0;
        width:100%;
        height:100%;
        background-color: rgba(0,0,0,0.7);
    }}
    </style>
""", unsafe_allow_html=True)

# Load Models
models = {
    'diabetes': pickle.load(open('diabetes_model.pkl', 'rb')),
    'parkinsons': pickle.load(open('parkinsons_model.pkl', 'rb'))
}

# Dropdown menu
selected = st.selectbox(
    'Select a Disease to Predict',
    ['Diabetes', 'Parkinsons Disease']
)

def display_input(label, key):
    return st.number_input(label, key=key, step=0.01)

if selected == 'Diabetes':
    st.title('Diabetes Prediction')
    st.write("Enter the following details to predict diabetes:")

    Pregnancies = display_input('Number of Pregnancies', 'pregnancies')
    Glucose = display_input("Glucose Level", 'glucose')
    BloodPressure = display_input("Blood Pressure", 'blood_pressure')
    SkinThickness = display_input("Skin Thickness", 'skin_thickness')
    Insulin = display_input("Insulin Level", 'insulin')
    BMI = display_input("BMI", 'bmi')
    DiabetesPedigreeFunction = display_input("Diabetes Pedigree Function", 'dpf')
    Age = display_input("Age", 'age')

    if st.button('Diabetes Test Result'):
        diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        st.success(result)

elif selected == 'Parkinsons Disease':
    st.title("Parkinson's Disease Prediction")
    st.write("Enter the following details to predict Parkinson's disease:")

    Jitter = display_input("Jitter (%)", 'jitter')
    Shimmer = display_input("Shimmer (dB)", 'shimmer')
    HNR = display_input("Harmonic-to-Noise Ratio (HNR)", 'hnr')

    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = models['parkinsons'].predict([[Jitter, Shimmer, HNR]])
        result = "Parkinson's Disease Detected" if parkinsons_prediction[0] == 1 else "No Parkinson's Detected"
        st.success(result)
