import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Connect to the MongoDB cluster to store the inputs and the prediction
uri = "mongodb+srv://anurag:07121998@cluster0.ugo9l.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student']  # Create a new database
collection = db['student_pred'] # Create a new collection in the database

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
        st.success(f"Your prediction result is: {round(float(prediction[0]), 3)}")

        user_data["prediction"] = round(float(prediction[0]), 3)    # Add the prediction to the user_data dictionary
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, float) else value for key, value in user_data.items()}    # Convert the values to int or float if they are of type np.integer or np.float
        collection.insert_one(user_data)    # Insert the user_data dictionary to the MongoDB collection
    
if __name__ == "__main__":
    main()