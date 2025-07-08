import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
model=load_model('model.h5')
with open('l.pkl','rb') as file:
    label_encoder=pickle.load(file)
with open('ohe.pkl','rb') as file:
    ohe_encoder=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title('Customer Churn Prediction')
geography=st.selectbox('Geography',ohe_encoder.categories_[0])
gender=st.selectbox('Gender',label_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0.10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_Active_member=st.selectbox('Is Active Member',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_Active_member],
    'EstimatedSalary':[estimated_salary]
})
geo_encoded=ohe_encoder.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe_encoder.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_Scaled=scaler.transform(input_data)

prediction=model.predict(input_data_Scaled)
prediction_proba=prediction[0][0]

if(prediction_proba>0.5):
    st.write('The Customer is Likely to Churn.')
else:
    st.write('The Customer is not likely to Churn.')