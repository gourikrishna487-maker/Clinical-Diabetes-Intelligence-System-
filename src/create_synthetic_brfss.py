import pandas as pd
import numpy as np
import os

def create_synthetic_brfss(output_path='data/brfss_raw.csv'):
    np.random.seed(42)
    rows = 10000
    
    # We will generate a base "risk score" to make features correlated
    base_risk = np.random.normal(0, 1, size=rows)
    
    data = {}
    
    # Age (1 to 13), correlates with risk
    age = np.clip(np.round((base_risk * 2) + 7), 1, 13)
    data['Age'] = age.astype(int)
    
    # BMI (15 to 60), correlates with risk
    bmi = np.clip(np.random.normal(25, 4, size=rows) + (base_risk * 3), 15, 60)
    data['BMI'] = bmi.astype(int)
    
    # HighBP
    bp_prob = 1 / (1 + np.exp(-(base_risk + (age/10) - 1)))
    data['HighBP'] = np.random.binomial(1, bp_prob)
    
    # HighChol
    chol_prob = 1 / (1 + np.exp(-(base_risk + (age/12) - 1.2)))
    data['HighChol'] = np.random.binomial(1, chol_prob)
    
    # GenHlth (1 to 5, 5 is worse)
    gen_hlth = np.clip(np.round((base_risk + 2.5) + np.random.normal(0, 0.5, size=rows)), 1, 5)
    data['GenHlth'] = gen_hlth.astype(int)
    
    # Other binary features
    data['CholCheck'] = np.random.choice([0, 1], size=rows, p=[0.05, 0.95])
    data['Smoker'] = np.random.binomial(1, 1 / (1 + np.exp(-(base_risk*0.5 - 0.5))))
    data['Stroke'] = np.random.binomial(1, 1 / (1 + np.exp(-(base_risk*0.8 - 2.5))))
    data['HeartDiseaseorAttack'] = np.random.binomial(1, 1 / (1 + np.exp(-(base_risk*0.8 - 2.0))))
    data['PhysActivity'] = np.random.binomial(1, np.clip(0.8 - base_risk*0.1, 0.1, 0.9))
    data['Fruits'] = np.random.choice([0, 1], size=rows, p=[0.4, 0.6])
    data['Veggies'] = np.random.choice([0, 1], size=rows, p=[0.2, 0.8])
    data['HvyAlcoholConsump'] = np.random.choice([0, 1], size=rows, p=[0.9, 0.1])
    data['AnyHealthcare'] = np.random.choice([0, 1], size=rows, p=[0.05, 0.95])
    data['NoDocbcCost'] = np.random.choice([0, 1], size=rows, p=[0.9, 0.1])
    data['MentHlth'] = np.clip(np.round(np.abs(np.random.normal(2, 5, size=rows))), 0, 30).astype(int)
    data['PhysHlth'] = np.clip(np.round((gen_hlth - 1) * 5 + np.random.normal(0, 5, size=rows)), 0, 30).astype(int)
    data['DiffWalk'] = np.random.binomial(1, 1 / (1 + np.exp(-(base_risk*0.6 + (age/10) - 2))))
    data['Sex'] = np.random.choice([0, 1], size=rows)
    data['Education'] = np.random.choice([1, 2, 3, 4, 5, 6], size=rows, p=[0.01, 0.04, 0.15, 0.3, 0.2, 0.3])
    data['Income'] = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=rows)
    
    # Target: Diabetes_binary
    # Higher risk, higher age, higher BMI -> much higher chance of diabetes
    # Specifically formula using standard logistic function
    diabetes_logit = (base_risk * 1.5) + (data['HighBP'] * 1.2) + (data['HighChol'] * 0.8) + \
                     ((data['BMI'] - 25) * 0.1) + ((data['Age'] - 6) * 0.2) + \
                     (data['GenHlth'] * 0.3) - 3.5
    
    diabetes_prob = 1 / (1 + np.exp(-diabetes_logit))
    data['Diabetes_binary'] = np.random.binomial(1, diabetes_prob)
    
    df = pd.DataFrame(data)
    
    # Ensure correct column order matching original BRFSS
    cols = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
            'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth',
            'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    df = df[cols]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Created synthetic realistic BRFSS dataset at {output_path} with {rows} rows.")
    print(f"Diabetes prevalence: {df['Diabetes_binary'].mean():.2%}")

if __name__ == "__main__":
    create_synthetic_brfss()
