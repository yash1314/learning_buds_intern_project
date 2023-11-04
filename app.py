import streamlit as st
from src.exception import CustomException 
import src.utils as utils
from src.pipelines.prediction_pipeline import PredictPipeline
import time
import sys

predict = PredictPipeline()

st.title('*Diabetes Prediction*')
# st.markdown("-------------------")
st.markdown("**Note**: This project is in 'NO' means an actual way of testing a diabetes disease. This project is trained on past diabetes data and does not reflect actual testing by any means.")
st.markdown("-------------------")

st.markdown(" ")

cols1, cols2 = st.columns(2)

with cols1:
    pregnancies = st.text_input('Pregnancies: ')
with cols2:
    glucose = st.text_input('Glucose:', placeholder='Range: 70 - 200 mg/dL')

with cols1:
    bloodpressure = st.text_input('Bloodpressure:',  placeholder='Range: 90/60 - 120/80 mm')
with cols2:
    insulin = st.text_input('Insulin:', placeholder='Range: 2 - 30 ulU/mL')

with cols1:
    bmi = st.text_input('Bmi:', placeholder='Range: 18-35 bmi')
with cols2:
    diabetespedigreefunction = st.text_input('Heredity:', placeholder='Range: 0-2.5')

with cols1:
    age = st.text_input('Age: ')
with cols2:
    

    st.markdown(' ')
    st.markdown(' ')
    
    if st.button("Predict", use_container_width = True):

        user_input = [pregnancies, glucose, bloodpressure, insulin, bmi, diabetespedigreefunction, age]
        
        if any(value == '' for value in user_input):
            st.error("Please enter values for all input fields.")

        else:
            user_input = [float(i) for i in user_input]

            result = predict.predict(features=user_input)

            with st.spinner('Processing!'):
                time.sleep(1)

            # Display prediction
            if result == 1:
                st.success("Diabetes")
            else:
                st.error("No diabetes")


        
        
