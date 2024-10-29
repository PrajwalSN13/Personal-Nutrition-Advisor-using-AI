import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Specify the file path to your dataset
file_path = r'D:\4 ARCHIVES\HBO Workbook - College\SIT\SIT Hackathon 2\advisor_6\backend\large_nutrition_chronic_kidney_disease.csv'

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    
    label_encoders = {}
    categorical_columns = ['Chronic Kidney Disease', 'Diabetes', 'Physical Activity Frequency', 'Allergy']
    
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].str.strip().str.capitalize())
        label_encoders[column] = le

    return data, label_encoders

# Call the function to load and process the data
if __name__ == "__main__":
    data, _ = load_and_process_data(file_path)
    print(data.head())  # Display the first few rows of the processed data to verify
