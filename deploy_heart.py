# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:59:00 2022

@author: Alfiqmal
"""
import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
physical_devices = tf.config.list_physical_devices("CPU")

#%% PATH

DATA_PATH = os.path.join(os.getcwd(), "heart.csv")
MMS_PATH = os.path.join(os.getcwd(), "scaler and model", "mms_scaler.pkl")
MODEL_PATH = os.path.join(os.getcwd(), "scaler and model", "model.pkl" )

#%% Model Loading

model_deploy = pickle.load(open(MODEL_PATH, "rb"))
scaler_deploy = pickle.load(open(MMS_PATH, "rb"))

heart_risk = {0: "Less chance of Heart Attack",
              1: "High chance of Heart Attack"}



#%% Streamlit

with st.form("Heart Attack Analysis and Prediction Form"):
    st.write("Personal Medical Information")
    age = st.number_input("Your age ")
    sex = st.number_input("State your sex (0 - Male, 1 - Female)")
    cp = st.number_input("State the type of your chest pain \
                             (0 = Typical Angina, 1 = Atypical Angina, \
                              2 = Non-anginal Pain, 3 = Asymptomatic)")
    trtbps = st.number_input("State your resting blood pressure (mm Hg)")
    chol = st.number_input("State your cholestrol level (mm/dl)")
    fbs = st.number_input("Your fasting blood sugar is > 120 mg/dl?\
                              (1 - True, 0 - False)")
    restecg = st.number_input("State your resting ECG result \
                                  (0 = Normal, 1 = ST-T wave normality, \
                                   2 = Left ventricular hypertrophy)")
    thalachh = st.number_input("State your Maximum heart rate achieved")
    exng = st.number_input("State your Exercise Induced Angina\
                               (1 - Yes, 0 - No)")
    oldpeak = st.number_input("State your previous peak ")
    slp = st.number_input("State your slope ")
    caa = st.number_input("State the number your of major vessels ")
    thall = st.number_input("State your Thalium Stress Test result ~ (0,3)")
    
    submitted = st.form_submit_button("Submit")
    
    if submitted == True:
        personal_med_info = np.array([age, sex, cp, trtbps, chol, fbs, restecg, 
                                      thalachh, exng, oldpeak, slp, caa, thall])
        personal_med_info = scaler_deploy.fit_transform(np.expand_dims(personal_med_info, 
                                                    axis=0))
           
        outcome = model_deploy.predict(personal_med_info)
        
        st.write(heart_risk[(np.argmax(outcome))])
        
        if np.argmax(outcome) == 1:
            st.warning(" Take care of your health, high risk of heart attack")
        else:
            st.success("Good job! low risk of heart attack!")
        
    st.write(submitted)
    
                              