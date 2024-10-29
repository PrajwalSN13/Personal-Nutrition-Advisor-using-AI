import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample data
data = {
    'Physical Activity Frequency': ['low', 'moderate', 'high', 'low'],
    'Chronic Kidney Disease': ['No', 'Yes', 'No', 'Yes'],
    'Diabetes': ['No', 'Yes', 'No', 'No'],
    'Allergy': ['No', 'No', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Initialize label encoders
label_encoders = {
    'Physical Activity Frequency': LabelEncoder(),
    'Chronic Kidney Disease': LabelEncoder(),
    'Diabetes': LabelEncoder(),
    'Allergy': LabelEncoder()
}

# Fit the encoders
for feature in label_encoders.keys():
    label_encoders[feature].fit(df[feature].str.strip().str.capitalize())

# Save the label encoders after fitting
joblib.dump(label_encoders['Physical Activity Frequency'], 'label_encoder_physical_activity.pkl')
joblib.dump(label_encoders['Chronic Kidney Disease'], 'label_encoder_chronic_kidney_disease.pkl')
joblib.dump(label_encoders['Diabetes'], 'label_encoder_diabetes.pkl')
joblib.dump(label_encoders['Allergy'], 'label_encoder_allergy.pkl')

print("Label encoders saved successfully.")
