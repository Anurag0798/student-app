import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_model():
    with open("student_lr_final_model.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocesssing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocesssing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student's performance prediction")
    st.write("Enter the data to get a prediction for your performance")
    
    hour_sutdied = st.number_input("Hours Studied", min_value = 1, max_value = 10, value = 5)
    prvious_score = st.number_input("Previous Score", min_value = 40, max_value = 100, value = 70)
    extra = st.selectbox("Extra Curricular Activity", ['Yes', "No"])
    sleeping_hour = st.number_input("Sleeping Hours", min_value = 4, max_value = 10, value = 7)
    number_of_paper_solved = st.number_input("Number of question paper solved", min_value = 0, max_value = 10, value = 5)
    
    if st.button("Predict the score"):
        user_data = {
            "Hours Studied": hour_sutdied,
            "Previous Scores": prvious_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_paper_solved
        }

        prediction = predict_data(user_data)
        st.success(f"Your prediction result is: {prediction}")
    
if __name__ == "__main__":
    main()