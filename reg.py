import streamlit as st
import numpy as np 
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import time 



#chargement du modele
with open('reg.pkl' , 'rb') as file:
    model = pickle.load(file)

#Titre et mise en apge
st.set_page_config(page_title="predicteur de charges medicales")
st.title("Predicteur de charges medicales")
st.markdown("remplissez les champs suivants pour predire les charges medicales")

#ajout d'animations
with st.spinner('Chargement du modele...'):
    time.sleep(2)

#entree utilisateur
col1 , col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 30)
with col2:
    sex = st.selectbox("Sexe", ["Homme", "Femme"])

col3 , col4 = st.columns(2)
with col3:
    bmi = st.number_input("BMI(Indice de masse corporelle)", 10.0, 50.0, 25.0)
with col4:
    children = st.slider("Nombre d'enfants", 0, 5, 1)

col5 , col6 = st.columns(2)
with col5:
    smoker = st.selectbox("Fumeur", ["Oui", "Non"])
with col6:
    region = st.selectbox("Region", ["Nord-Est", "Nord-Ouest", "Sud-Est", "Sud-Ouest"])

#encodage
sex_encoded = 1 if sex == "Homme" else 0
smoker_encoded = 1 if smoker == "Oui" else 0
region_dict = {"Nord-Est":0.24308153 , "Nord-Ouest":0.24442709 , "Sud-Est":0.24677265 , "Sud-Ouest":0.26569973  }
region_encoded = region_dict[region]

#preparation des données pour la prédiction
input_data  = [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]]

#prédiction
if st.button("Predire les charges medicales"):
    with st.spinner('Prediction en cours...'):
        prediction = model.predict(input_data)[0]
        time.sleep(2)   
    st.success("Prediction terminée!")
    st.markdown(f"### Les charges medicales Estimées : **${prediction:,.2f}**")
    st.balloons()

