import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_data(filepath):
    """
    Preprocess the input dataset: handles missing values, encodes categorical variables,
    scales features, and drops unnecessary columns.
    """
    data = pd.read_csv(filepath)

    # Create target variable 'y'
    data['y'] = (data['loan_status'] == 'Charged Off').astype(int)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    if 'grade' in data.columns:
        data['grade'] = label_encoder.fit_transform(data['grade'])
    if 'home_ownership' in data.columns:
        ownership_dummies = pd.get_dummies(data['home_ownership'], prefix='home_ownership')
        data = pd.concat([data, ownership_dummies], axis=1).drop('home_ownership', axis=1)
    if 'application_type' in data.columns:
        data['application_type'] = data['application_type'].replace({'Individual': 0, 'Joint App': 1})

    # Replace emp_length values
    emp_length_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10
    }
    if 'emp_length' in data.columns:
        data['emp_length'] = data['emp_length'].replace(emp_length_map)

    # Drop unnecessary columns
    unnecessary_cols = ['id', 'member_id', 'mths_since_last_delinq', 'loan_status']
    data = data.drop(columns=[col for col in unnecessary_cols if col in data.columns], errors='ignore')

    # Handle missing values
    if 'emp_length' in data.columns:
        data['emp_length'].fillna(data['emp_length'].median(), inplace=True)
    data.dropna(inplace=True)

    # Scale numeric columns
    scaler = MinMaxScaler()
    scaled_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[scaled_cols] = scaler.fit_transform(data[scaled_cols])

    return data
