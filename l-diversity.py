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

def check_k_anonymity(df, quasi_identifiers, k):
    
    """Check if the dataset satisfies k-anonymity for the given quasi-identifiers."""
    groups = df.groupby(quasi_identifiers)
    for _, group in groups:
        if len(group) < k:
            return False
    return True

def check_l_diversity(df, quasi_identifiers, sensitive_attribute, l):
    """Check if the dataset satisfies l-diversity"""
    groups = df.groupby(quasi_identifiers)
    for _, group in groups:
        distinct_sensitive_vals = group[sensitive_attribute].nunique()
        if distinct_sensitive_vals < l:
            return False
    return True

def achieve_l_diversity_by_suppression(df, quasi_identifiers, sensitive_attribute, l):
    """Achieve l-diversity by suppressing the data by sensitive_attribute."""
    groups = df.groupby(quasi_identifiers)
    df_list = []
    for _, group in groups:
        distinct_sensitive_vals = group[sensitive_attribute].nunique()
        if distinct_sensitive_vals >= l:
            df_list.append(group)
        else:
            # Suppress groups not meeting l-diversity
            pass
    return pd.concat(df_list).reset_index(drop=True)

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

df['AgeGroup'] = df['Age'].apply(generalize_age)
df['ZipGroup'] = df['ZIP'].apply(suppress_zip)

df_k_anon = df[['AgeGroup', 'Occupation', 'ZipGroup', 'Disease']].copy()

print("\nAfter k-Anonymization (Generalized):")
print(df_k_anon)

quasi_identifiers = ['AgeGroup', 'Occupation', 'ZipGroup']
k = 2
is_k_anonymous = check_k_anonymity(df_k_anon, quasi_identifiers, k)
print(f"\nIs the dataset k-anonymous with k={k}? {is_k_anonymous}")

l = 2
is_l_diverse = check_l_diversity(df_k_anon, quasi_identifiers, 'Disease', l)
print(f"Is the dataset l-diverse with l={l}? {is_l_diverse}")

df_l_diverse = achieve_l_diversity_by_suppression(
    df_k_anon,
    quasi_identifiers,
    'Disease',
    l
)

is_l_diverse_after = check_l_diversity(df_l_diverse, quasi_identifiers, 'Disease', l)
print("\nDataset after attempting to enforce l-diversity:")
print(df_l_diverse)
print(f"Is the new dataset l-diverse with l={l}? {is_l_diverse_after}")
