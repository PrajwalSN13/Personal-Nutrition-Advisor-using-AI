import joblib
import pandas as pd

# Load label encoders from files
label_encoders = {
    'Physical Activity Frequency': joblib.load('label_encoder_physical_activity.pkl'),
    'Chronic Kidney Disease': joblib.load('label_encoder_chronic_kidney_disease.pkl'),
    'Diabetes': joblib.load('label_encoder_diabetes.pkl'),
    'Allergy': joblib.load('label_encoder_allergy.pkl'),
}

def predict_food_suggestions(age, weight, physical_activity_frequency, chronic_kidney_disease, diabetes, allergy):
    try:
        physical_activity_encoded = label_encoders['Physical Activity Frequency'].transform([physical_activity_frequency.strip().capitalize()])[0]
        chronic_kidney_disease_encoded = label_encoders['Chronic Kidney Disease'].transform([chronic_kidney_disease.strip().capitalize()])[0]
        diabetes_encoded = label_encoders['Diabetes'].transform([diabetes.strip().capitalize()])[0]
        allergy_encoded = label_encoders['Allergy'].transform([allergy.strip().capitalize()])[0]

        # Create a DataFrame for the user input
        user_data = pd.DataFrame([[age, weight, chronic_kidney_disease_encoded, physical_activity_encoded, diabetes_encoded, allergy_encoded]], 
                                  columns=['Age', 'Weight', 'Chronic Kidney Disease', 'Physical Activity Frequency', 'Diabetes', 'Allergy'])

        # Load the trained models
        model_can_eat = joblib.load('model_can_eat.pkl')
        model_cannot_eat = joblib.load('model_cannot_eat.pkl')

        # Make predictions
        can_eat_suggestion = model_can_eat.predict(user_data)
        cannot_eat_suggestion = model_cannot_eat.predict(user_data)

        return can_eat_suggestion.tolist(), cannot_eat_suggestion.tolist()

    except KeyError as e:
        print(f"Error: Unseen label {e}. Please ensure your inputs are correct.")
        return None, None

# Example usage
if __name__ == "__main__":
    age = 30
    weight = 70
    physical_activity_frequency = 'moderate'  # Example category
    chronic_kidney_disease = 'No'  # Use string 'Yes' or 'No'
    diabetes = 'No'  # Use string 'Yes' or 'No'
    allergy = 'No'  # Use string 'Yes' or 'No'

    can_eat, cannot_eat = predict_food_suggestions(age, weight, physical_activity_frequency, chronic_kidney_disease, diabetes, allergy)
    print(f"Can Eat: {can_eat}")
    print(f"Cannot Eat: {cannot_eat}")
