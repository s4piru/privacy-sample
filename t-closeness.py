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
    return "XXXXX"

def check_k_anonymity(df, quasi_identifiers, k):
    """Check if the dataset satisfies k-anonymity for the given quasi-identifiers."""
    groups = df.groupby(quasi_identifiers)
    for _, group in groups:
        if len(group) < k:
            return False
    return True

def check_l_diversity(df, quasi_identifiers, sensitive_attribute, l):
    """Check if the dataset satisfies l-diversity."""
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
    if df_list:
        return pd.concat(df_list).reset_index(drop=True)
    else:
        print("Warning: No groups meet l-diversity after suppression.")
        return pd.DataFrame(columns=df.columns)

def check_t_closeness(df, quasi_identifiers, sensitive_attribute, t):
    """
    Check if the dataset satisfies t-closeness.
    t-closeness requires that the distance between the distribution of the sensitive 
    attribute in each equivalence class (group) and the distribution of the sensitive 
    attribute in the entire dataset is less than or equal to t.

    Here, we'll use a simple metric: sum of absolute differences in proportions 
    across all sensitive attribute categories.
    """

    # Calculate global distribution of the sensitive attribute
    global_dist = df[sensitive_attribute].value_counts(normalize=True)

    # Group by the quasi-identifiers
    groups = df.groupby(quasi_identifiers)
    
    for _, group in groups:
        # Calculate local distribution of the sensitive attribute in this group
        local_dist = group[sensitive_attribute].value_counts(normalize=True)
        
        # Calculate the difference between local_dist and global_dist
        # We use sum of absolute differences for each category in global_dist.
        distance = 0.0
        for val in global_dist.index:
            global_prob = global_dist[val]
            local_prob = local_dist[val] if val in local_dist.index else 0
            distance += abs(global_prob - local_prob)
        
        # If the distance is greater than t, t-closeness is violated
        if distance > t:
            return False
    return True

def achieve_t_closeness_by_suppression(df, quasi_identifiers, sensitive_attribute, t):
    """
    Suppress entire groups that do not meet t-closeness. 
    That is, if a group's distribution of the sensitive attribute 
    differs from the global distribution by more than t, 
    we remove (suppress) that group entirely.
    """
    # Calculate global distribution of the sensitive attribute
    global_dist = df[sensitive_attribute].value_counts(normalize=True)

    groups = df.groupby(quasi_identifiers)
    df_list = []

    for _, group in groups:
        local_dist = group[sensitive_attribute].value_counts(normalize=True)
        
        # Calculate distance
        distance = 0.0
        for val in global_dist.index:
            global_prob = global_dist[val]
            local_prob = local_dist[val] if val in local_dist.index else 0
            distance += abs(global_prob - local_prob)

        # If distance <= t, keep the group; otherwise suppress (omit) it
        if distance <= t:
            df_list.append(group)

    if df_list:
        return pd.concat(df_list).reset_index(drop=True)
    else:
        print("Warning: No groups meet t-closeness after suppression.")
        return pd.DataFrame(columns=df.columns)

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

df_l_diverse = achieve_l_diversity_by_suppression(df_k_anon, quasi_identifiers, 'Disease', l)
is_l_diverse_after = check_l_diversity(df_l_diverse, quasi_identifiers, 'Disease', l)
print("\nDataset after attempting to enforce l-diversity:")
print(df_l_diverse)
print(f"Is the new dataset l-diverse with l={l}? {is_l_diverse_after}")

t = 0.4  
is_t_close = check_t_closeness(df_l_diverse, quasi_identifiers, 'Disease', t)
print(f"\nIs the dataset t-close with t={t}? {is_t_close}")

df_t_close = achieve_t_closeness_by_suppression(df_l_diverse, quasi_identifiers, 'Disease', t)
is_t_close_after = check_t_closeness(df_t_close, quasi_identifiers, 'Disease', t)
print("\nDataset after attempting to enforce t-closeness:")
print(df_t_close)
print(f"Is the new dataset t-close with t={t}? {is_t_close_after}")
