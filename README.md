# Diabetes Prediction and Recommendation System

This project is a comprehensive Machine Learning application that uses physical indicators and blood tests to assess the risk of diabetes in patients. It provides personalized health recommendations based on the predicted risk level.

## 📁 Project Structure
- `data/`: Contains raw and preprocessed Pima Indians Diabetes datasets.
- `models/`: Stores the trained Random Forest model and the feature scaler.
- `src/`: Core Python modules for preprocessing, training, prediction, and recommendations.
- `app/`: Streamlit web dashboard for user Interaction.
- `database/`: SQLite database for tracking assessment history.

## 🚀 Features
- **Data Preprocessing**: Handles logically impossible zero values and feature scaling.
- **Machine Learning**: Random Forest Classifier with ~75% accuracy.
- **Recommendation Engine**: Personalized health advice based on glucose levels, BMI, and overall risk.
- **Dashboard**: Interactive Streamlit interface with history tracking.

## 🛠️ How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app/app.py
   ```

## 📊 Dataset
The project uses the **Pima Indians Diabetes Dataset**, which includes several medical predictor variables and one target variable (Outcome).
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
