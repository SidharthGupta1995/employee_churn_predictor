# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle

# Load your trained churn model
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit App
st.title("Employee Churn Prediction")

uploaded_file = st.file_uploader("Upload Employee Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # Run prediction
    if st.button("Predict Churn"):
        predictions = model.predict(df)
        df['Churn_Prediction'] = predictions
        st.success("Churn predictions generated!")
        st.write("### Prediction Results", df[['Churn_Prediction']].value_counts())

        st.download_button(
            label="Download Results as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='churn_predictions.csv',
            mime='text/csv'
        )
