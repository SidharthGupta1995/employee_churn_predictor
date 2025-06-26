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

    # Define required columns as per model
    categorical_columns = ['Role','Gender','Shift_Type','Reason_for_Leaving',
                           'Job_Satisfaction_Category','Last_Increment_Bracket']
    numeric_columns = ['Age', 'Experience_Years','Distance_km', 'Salary_Hourly','Engagement_Score',
                       'Tenure_Months','Satisfaction_Level', 'Avg_Monthly_Hours','Work_Accident', 
                       'Promotion_Last_2Years','Salary_Change_%','Overtime_Hours_Week',
                       'Days_Since_Last_Leave','HR_Complaint', 'Conflict_Reported', 'Years_in_Current_Role']

    required_columns = categorical_columns + numeric_columns

    # Ensure categorical columns are strings and fill missing
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Missing")

    # Ensure numeric columns are numeric and clean
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only required columns in correct order
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
    else:
        df_model_input = df[required_columns].copy()
        df_model_input.dropna(inplace=True)

        # Run prediction
        if st.button("Predict Churn"):
            try:
                predictions = model.predict(df_model_input)
                df['Churn_Prediction'] = predictions
                st.success("Churn predictions generated!")
                st.write("### Prediction Results", df[['Churn_Prediction']].value_counts())

                st.download_button(
                    label="Download Results as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='churn_predictions.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
