import joblib
import pandas as pd
import numpy as np
import shap

class DiabetesPredictor:
    def __init__(self, mode='clinical'):
        self.mode = mode
        if mode == 'clinical':
            self.pipeline = joblib.load('models/clinical_pipeline.pkl')
            self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        else:
            self.pipeline = joblib.load('models/lifestyle_pipeline.pkl')
            self.feature_names = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
        
        try:
            # TreeExplainer expects the classifier specifically, not the whole pipeline
            classifier_step = self.pipeline.named_steps['classifier']
            self.explainer = shap.TreeExplainer(classifier_step)
        except Exception as e:
            print(f"SHAP Explainer initialization error: {e}")
            self.explainer = None

    def predict(self, data_dict):
        # Ensure all features are present in the correct order
        input_df = pd.DataFrame([data_dict], columns=self.feature_names)
        
        # Predict probability
        prediction_prob = self.pipeline.predict_proba(input_df)[0][1]
        prediction_class = self.pipeline.predict(input_df)[0]
        
        return {
            'prediction_class': int(prediction_class),
            'probability': float(prediction_prob),
            'input_df': input_df
        }

    def get_explanations(self, input_df):
        """Returns feature importance (SHAP values) for a single sample."""
        if self.explainer:
            # To get SHAP values, we must pass the data through all prior pipeline steps
            transformed_input = input_df
            for name, step in self.pipeline.named_steps.items():
                if name == 'classifier':
                    break
                transformed_input = step.transform(transformed_input)
                
            shap_values = self.explainer.shap_values(transformed_input)
            
            # Handle different versions of SHAP output for binary classification
            if isinstance(shap_values, list):
                val = shap_values[1][0] # Class 1
            elif len(shap_values.shape) == 3:
                val = shap_values[0, :, 1]
            else:
                val = shap_values[0]
            
            return dict(zip(self.feature_names, val))
        return None

if __name__ == "__main__":
    # Test Clinical
    p_clin = DiabetesPredictor(mode='clinical')
    sample_clin = {'Pregnancies': 0, 'Glucose': 120, 'BloodPressure': 70, 'SkinThickness': 20, 'Insulin': 80, 'BMI': 25, 'DiabetesPedigreeFunction': 0.5, 'Age': 30}
    print("Clinical Prediction:", p_clin.predict(sample_clin))
    
    # Test Lifestyle
    p_life = DiabetesPredictor(mode='lifestyle')
    sample_life = {f: 0 for f in p_life.feature_names}
    sample_life['BMI'] = 35
    sample_life['HighBP'] = 1
    sample_life['GenHlth'] = 4
    sample_life['Age'] = 12
    print("Lifestyle Prediction:", p_life.predict(sample_life))
