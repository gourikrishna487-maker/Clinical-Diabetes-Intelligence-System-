import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

def train_clinical_model(raw_path, model_path):
    print("Training Clinical Model...")
    df = pd.read_csv(raw_path)
    
    # Replace 0 with NaN for columns where 0 is logically impossible
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Clinical Model Accuracy: {accuracy:.2%}")
    joblib.dump(pipeline, model_path)

def train_lifestyle_model(raw_path, model_path):
    print("Training Lifestyle Model...")
    df = pd.read_csv(raw_path)
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Lifestyle Model Accuracy: {accuracy:.2%}")
    joblib.dump(pipeline, model_path)

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_clinical_model('data/raw_data.csv', 'models/clinical_pipeline.pkl')
    train_lifestyle_model('data/brfss_raw.csv', 'models/lifestyle_pipeline.pkl')
