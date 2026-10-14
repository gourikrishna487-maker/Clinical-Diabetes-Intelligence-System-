class DiabetesRecommendationEngine:
    def __init__(self, thresholds=(0.1, 0.4)):
        self.thresholds = thresholds  # Low Risk: [0, 0.1), Medium Risk: [0.1, 0.4), High Risk: [0.4, 1.0]

    def get_recommendations(self, probability, data_dict):
        risk_level = self.get_risk_level(probability)
        
        recs = {
            'risk_level': risk_level,
            'advice': self._generate_advice_list(risk_level, data_dict)
        }
        return recs

    def get_risk_level(self, probability):
        if probability < self.thresholds[0]:
            return "Low"
        elif probability < self.thresholds[1]:
            return "Medium"
        else:
            return "High"

    def _generate_advice_list(self, risk_level, data_dict):
        advice = []
        
        # Base Clinical Thresholds (Scientific Context)
        if risk_level == "High":
            advice.append({
                "tip": "⚠️ Immediate Clinical Consultation Required",
                "detail": "Your current bio-profile indicates a high risk of metabolic imbalance. This means your body may be struggling with insulin resistance, leading to persistent high blood sugar levels that can damage organs over time."
            })
            advice.append({
                "tip": "🩺 Intense Glucose Monitoring",
                "detail": "Regularly tracking glucose helps you understand 'glycemic variability'—the spikes and dips in your sugar. High variability is a major risk factor for cardiovascular complications."
            })
        elif risk_level == "Medium":
            advice.append({
                "tip": "🔍 Strategic Pre-Diabetes Screening",
                "detail": "At this level, you might be in a 'pre-diabetic' phase where lifestyle changes are most effective. Early intervention can often reverse the trend before it becomes chronic."
            })
            advice.append({
                "tip": "🍏 Metabolic-Friendly Dieting",
                "detail": "Reduce high-glycemic index foods (white bread, sugar) which cause rapid insulin spikes. Focus on 'Complex Carbs' that digest slowly and provide steady energy."
            })
        else:
            advice.append({
                "tip": "✅ Protective Maintenance",
                "detail": "Your risk score is optimal, but consistency is key. Maintaining low inflammation through diet and sleep keeps your metabolic flexibility high."
            })

        # Feature-Specific Scientific Context
        if 'BMI' in data_dict and data_dict['BMI'] > 25:
            advice.append({
                "tip": "⚖️ Adipose Tissue & Insulin Sensitivity",
                "detail": f"A BMI of {data_dict['BMI']:.1f} indicates excess adipose tissue (body fat). Fat cells, particularly around the abdomen, produce inflammatory markers that make your body less responsive to insulin."
            })
        
        if 'Glucose' in data_dict and data_dict['Glucose'] > 140:
            advice.append({
                "tip": "🍭 Postprandial Glucose Surge",
                "detail": "A glucose level above 140 suggests your pancreas is challenged when clearing sugar from your bloodstream. Over time, high sugar 'glazes' your proteins (glycation), leading to tissue damage."
            })

        # Lifestyle Habits (Scientific Pathify)
        if data_dict.get('Smoker') == 1:
            advice.append({
                "tip": "🚭 Nicotine & Cortisol",
                "detail": "Smoking increases cortisol levels, a stress hormone that tells the liver to release more glucose, directly worsening insulin resistance."
            })
        
        if data_dict.get('PhysActivity') == 0:
            advice.append({
                "tip": "🚶 Mitochondrial Biogenesis",
                "detail": "Muscles are the primary consumers of glucose. Inactivity causes mitochondria (your cells' powerhouses) to become less efficient at burning sugar."
            })

        if data_dict.get('Fruits') == 0 or data_dict.get('Veggies') == 0:
            advice.append({
                "tip": "🥦 Micronutrient Deficiency",
                "detail": "Leafy greens provide magnesium and chromium, essential co-factors for insulin signaling. Fiber also slows down sugar absorption in the gut."
            })

        return advice
