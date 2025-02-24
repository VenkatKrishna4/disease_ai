import streamlit as st
import pickle
from streamlit_option_menu import option_menu

st.set_page_config(page_title = "Disease Prediction",page_icon="⚕")

#hiding streamlit add-ons
hide_st_style = """
<style>
#MainMenu{visibility: hidden;}
footer{visibility: hidden;}
header{visibility: hidden;}
</style>
"""

background_image_url ="https://www.shutterstock.com/image-illustration/biomarker-discovery-diagnostic-prognostic-predictive-600nw-2197072529.jpg"


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({background_image_url});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stAppViewContainer"]::before{{
content:"";
position: absolute;
top:0;
left:0;
width:100%;
height:100%;
background-color: rgba(0,0,0,0.7);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

models={
    'diabetes': pickle.load(open('diabetes_model.sav','rb')),
    'parkinsons': pickle.load(open('parkinsons_model.sav','rb'))
}

#dropdown menu

selected = st.selectbox (
    'Select a Disease to Predict',
    ['Diabetes','Parkinsons Disease']
)

def display_input(label,tooltip, key, type="text"):
    if type == "text":
        return st.text_input(label,key=key,help=tooltip)
    elif type == "number":
        return st.number_input(label, key=key, help=tooltip,step=1)


if selected =='Diabetes':
    st.title('Diabetes')
    st.write("Enter the following details to predict diabetes:")

    Pregnancies = display_input('No of Pregnancies','number')
    Glucose = display_input("Glucose Level",'number')
    BloodPressure = display_input('number')
    SkinThickness = display_input('number')
    Insulin = display_input('number')
    BMI = display_input('number')
    DiabetesPedigreeFunction = display_input('number')
    Age = display_input('number')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        diab_prediction = models['diabetes'].predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        diab_diagnosis= 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        st.success(diab_diagnosis)
