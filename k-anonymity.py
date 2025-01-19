import pandas as pd

def generalize_age(age):
    """Generalize age into predefined age ranges."""
    if 20 <= age < 30:
        return "20-29"
    elif 30 <= age < 40:
        return "30-39"
    elif 40 <= age < 50:
        return "40-49"
    else:
        return "50+"

def suppress_zip(zip_code):
    """Suppress ZIP code by replacing digits with 'XXXXX'."""
    zip_str = str(zip_code)
    return "XXXXX"

def generalize_occupation(occupation):
    """Generalize occupation into broader categories."""
    technical_occupations = {
        'Engineer', 'Doctor', 'Teacher', 'Nurse', 'Lawyer',
        'Accountant', 'Scientist', 'Pharmacist', 'Technician'
    }
    if occupation in technical_occupations:
        return 'Technical'
    else:
        return 'Other'

def check_k_anonymity(df, quasi_identifiers, k):
    """
    Check if the dataset satisfies k-anonymity for the given quasi-identifiers.

    Parameters:
    - df: pandas DataFrame
    - quasi_identifiers: list of column names to consider as quasi-identifiers
    - k: the anonymity parameter

    Returns:
    - True if k-anonymity is satisfied, False otherwise
    """
    group_counts = df.groupby(quasi_identifiers).size()
    return (group_counts >= k).all()

def achieve_k_anonymity(df, quasi_identifiers):
    """
    Achieve k-anonymity by generalizing quasi-identifiers.

    Parameters:
    - df: pandas DataFrame
    - quasi_identifiers: list of column names to consider as quasi-identifiers

    Returns:
    - A new DataFrame that satisfies k-anonymity
    """
    df_generalized = df.copy()
    
    # Generalize Age
    if 'Age' in quasi_identifiers:
        df_generalized['Age'] = df_generalized['Age'].apply(generalize_age)
    
    # Suppress ZIP Code
    if 'ZIP' in quasi_identifiers:
        df_generalized['ZIP'] = df_generalized['ZIP'].apply(suppress_zip)
    
    # Generalize Occupation
    if 'Occupation' in quasi_identifiers:
        df_generalized['Occupation'] = df_generalized['Occupation'].apply(generalize_occupation)
    
    return df_generalized

# Sample dataset
data = [
    {'Age': 20, 'Occupation': 'Engineer', 'ZIP': '12345', 'Disease': 'Flu'},
    {'Age': 24, 'Occupation': 'Engineer', 'ZIP': '12345', 'Disease': 'Flu'},
    {'Age': 25, 'Occupation': 'Teacher', 'ZIP': '12345', 'Disease': 'Diabetes'},
    {'Age': 28, 'Occupation': 'Teacher', 'ZIP': '12345', 'Disease': 'Cancer'},
    {'Age': 40, 'Occupation': 'Engineer', 'ZIP': '12345', 'Disease': 'Flu'},
    {'Age': 45, 'Occupation': 'Engineer', 'ZIP': '12345', 'Disease': 'Diabetes'},
    {'Age': 48, 'Occupation': 'Teacher', 'ZIP': '12345', 'Disease': 'Cancer'},
    {'Age': 49, 'Occupation': 'Teacher', 'ZIP': '12345', 'Disease': 'Cancer'},
]

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# Define quasi-identifiers
quasi_identifiers = ['Age', 'Occupation', 'ZIP']
k = 3

# Check k-anonymity before generalization
is_k_anonymous = check_k_anonymity(df, quasi_identifiers, k)
print(f"\nIs the original dataset {k}-anonymous? {is_k_anonymous}")

# Achieve k-anonymity by generalizing
df_anonymized = achieve_k_anonymity(df, quasi_identifiers)
print("\nAnonymized Data:")
print(df_anonymized)

# Check k-anonymity after generalization
is_k_anonymous_after = check_k_anonymity(df_anonymized, quasi_identifiers, k)
print(f"\nIs the anonymized dataset {k}-anonymous? {is_k_anonymous_after}")
