import sys 
import os 
sys.path.append(os.path.abspath(".."))
import streamlit as st
import pickle
import pandas as pd
from utils.preprocess import titanic_preprocess

model = pickle.load(open("models/titanic_pipeline.pkl", "rb"))


st.set_page_config(
    page_title="ML Project - Titanic",
    page_icon="🐸",
)
st.title("Titanic Survival Prediction")

pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("SibSp", 0, 8, 0)
parch = st.slider("Parch", 0, 6, 0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

if st.button("Predict"):
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    survival_prob = proba[0][1]
    death_prob = proba[0][0]

    st.subheader("Prediction Result")
    st.dataframe(input_df)
    if prediction[0] == 1:
        st.success("Survived")
    else:
        st.error("Did not survive")

    
    st.write(f"Survival Probability: {survival_prob:.2%}")
    st.progress(float(survival_prob))

    st.write(f"Death Probability: {death_prob:.2%}")
    st.progress(float(death_prob))