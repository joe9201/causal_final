import pandas as pd
import numpy as np

def load_and_prepare_student_data(file_path):
    df = pd.read_csv(file_path)
    df['G_avg'] = df[['G1', 'G2', 'G3']].mean(axis=1)
    
    columns_to_keep = ['absences', 'failures', 'internet', 'higher', 'Medu', 'health', 'famsup', 'Pstatus', 'famrel', 'schoolsup', 'G_avg', 'paid', 'studytime']
    df_filtered = df[columns_to_keep]
    
    df_encoded = pd.get_dummies(df_filtered, columns=['internet', 'higher', 'famsup', 'paid'], drop_first=True)
    
    ordinal_map = {
        'Pstatus': {'A': 1, 'T': 2},
        'famrel': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        'schoolsup': {'no': 0, 'yes': 1},
        'health': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    }
    
    for col, mapping in ordinal_map.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
    
    df_encoded.dropna(inplace=True)  # Drop rows with any NaN values
    labels = df_encoded.columns.tolist()
    data = df_encoded.to_numpy()
    
    return df_encoded, labels, data

def load_and_prepare_adult_data(file_path):
    df = pd.read_csv(file_path)
    
    columns_to_keep = ['age', 'workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'hours.per.week', 'native.country', 'income']
    df_filtered = df[columns_to_keep]

    # Ordinal encoding mappings
    workclass_order = {'Private': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3, 'Federal-gov': 4, 'Local-gov': 5, 'State-gov': 6, 'Without-pay': 7, 'Never-worked': 8}
    education_order = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-acdm': 11, 'Assoc-voc': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}
    marital_status_order = {'Married-civ-spouse': 1, 'Divorced': 2, 'Never-married': 3, 'Separated': 4, 'Widowed': 5, 'Married-spouse-absent': 6, 'Married-AF-spouse': 7}
    occupation_order = {'Tech-support': 1, 'Craft-repair': 2, 'Other-service': 3, 'Sales': 4, 'Exec-managerial': 5, 'Prof-specialty': 6, 'Handlers-cleaners': 7, 'Machine-op-inspct': 8, 'Adm-clerical': 9, 'Farming-fishing': 10, 'Transport-moving': 11, 'Priv-house-serv': 12, 'Protective-serv': 13, 'Armed-Forces': 14}
    relationship_order = {'Wife': 1, 'Own-child': 2, 'Husband': 3, 'Not-in-family': 4, 'Other-relative': 5, 'Unmarried': 6}
    race_order = {'White': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4, 'Black': 5}
    sex_order = {'Male': 1, 'Female': 2}
    native_country_order = {'United-States': 1, 'Cambodia': 2, 'England': 3, 'Puerto-Rico': 4, 'Canada': 5, 'Germany': 6, 'Outlying-US(Guam-USVI-etc)': 7, 'India': 8, 'Japan': 9, 'Greece': 10, 'South': 11, 'China': 12, 'Cuba': 13, 'Iran': 14, 'Honduras': 15, 'Philippines': 16, 'Italy': 17, 'Poland': 18, 'Jamaica': 19, 'Vietnam': 20, 'Mexico': 21, 'Portugal': 22, 'Ireland': 23, 'France': 24, 'Dominican-Republic': 25, 'Laos': 26, 'Ecuador': 27, 'Taiwan': 28, 'Haiti': 29, 'Columbia': 30, 'Hungary': 31, 'Guatemala': 32, 'Nicaragua': 33, 'Scotland': 34, 'Thailand': 35, 'Yugoslavia': 36, 'El-Salvador': 37, 'Trinadad&Tobago': 38, 'Peru': 39, 'Hong': 40, 'Holand-Netherlands': 41}
    
    ordinal_map = {
        'workclass': workclass_order,
        'education': education_order,
        'marital.status': marital_status_order,
        'occupation': occupation_order,
        'relationship': relationship_order,
        'race': race_order,
        'sex': sex_order,
        'native.country': native_country_order,
        'income': {'<=50K': 0, '>50K': 1}
    }

    for col, mapping in ordinal_map.items():
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].map(mapping)
    
    df_encoded = df_filtered * 1

    # Check for non-numeric values and NaNs
    print("Adult Data Preparation:")
    print(df_encoded.dtypes)
    print("Missing values:\n", df_encoded.isnull().sum())

    # Ensure all values are numeric
    assert np.issubdtype(df_encoded.values.dtype, np.number), "Data contains non-numeric values"
    assert not df_encoded.isnull().values.any(), "Data contains NaNs"

    labels = df_encoded.columns.tolist()
    data = df_encoded.to_numpy()
    
    return df_encoded, labels, data

# Variable mapping for student dataset
student_variable_mapping = {
    'internet': 'internet_yes',
    'famsup': 'famsup_yes',
    'higher': 'higher_yes',
    'paid': 'paid_yes'
}

def apply_variable_mapping(variables, mapping):
    return [mapping.get(var, var) for var in variables]
