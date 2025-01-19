import copy
import numpy as np
import math
from collections import Counter

def laplace_noise(scale):
    return np.random.laplace(0, scale)

def apply_laplace_mechanism(data, attribute, epsilon, sensitivity=1.0):
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    
    scale = sensitivity / epsilon
    sanitized_data = copy.deepcopy(data)
    
    for record in sanitized_data:
        if attribute in record and isinstance(record[attribute], (int, float)):
            original_value = record[attribute]
            noise = laplace_noise(scale)
            noisy_value = original_value + noise
            record[attribute] = noisy_value
    
    return sanitized_data

def exponential_mechanism_selection(utilities, epsilon, sensitivity=1.0):
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    
    exp_scores = {}
    for category, utility in utilities.items():
        exp_score = math.exp((epsilon * utility) / (2 * sensitivity))
        exp_scores[category] = exp_score
    
    total = sum(exp_scores.values())
    probabilities = [exp_scores[cat] / total for cat in utilities.keys()]
    categories = list(utilities.keys())
    
    selected = np.random.choice(categories, p=probabilities)
    
    return selected

def apply_exponential_mechanism_individual(data, attribute, epsilon, sensitivity=1.0):
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    
    sanitized_data = copy.deepcopy(data)
    n = len(sanitized_data)
    
    epsilon_per_record = epsilon / n if n > 0 else 0
    
    unique_categories = list({record[attribute] for record in sanitized_data if attribute in record})
    
    for record in sanitized_data:
        if attribute in record:
            actual_category = record[attribute]
            utilities = {category: 1 if category == actual_category else 0 for category in unique_categories}
            
            selected_category = exponential_mechanism_selection(utilities, epsilon_per_record, sensitivity)
            record[attribute] = selected_category
    
    return sanitized_data

def apply_differential_privacy(data, epsilon_age=1.0, epsilon_disease=1.0):
    data_with_noisy_age = apply_laplace_mechanism(data, 'Age', epsilon_age, sensitivity=1.0)
    
    data_with_noisy_disease = apply_exponential_mechanism_individual(
        data_with_noisy_age, 'Disease', epsilon_disease, sensitivity=1.0
    )
    
    return data_with_noisy_disease

def dp_privacy_check(original_data, dp_function, epsilon, attribute='Disease', trials=100):
    neighbor_data = original_data[:-1]
    
    def count_category_in_sanitized(sanitized, category='Flu'):
        return sum(1 for r in sanitized if r.get(attribute) == category)
    
    original_counts = []
    neighbor_counts = []
    for _ in range(trials):
        dp_result_original = dp_function(original_data)
        dp_result_neighbor = dp_function(neighbor_data)
        original_counts.append(count_category_in_sanitized(dp_result_original, 'Flu'))
        neighbor_counts.append(count_category_in_sanitized(dp_result_neighbor, 'Flu'))
    
    original_dist = Counter(original_counts)
    neighbor_dist = Counter(neighbor_counts)
    
    original_dist = {k: v / trials for k, v in original_dist.items()}
    neighbor_dist = {k: v / trials for k, v in neighbor_dist.items()}
    
    print(f"DP Privacy Check (Attribute: {attribute})")
    print("Distribution of 'Flu' counts in sanitized data:")
    print("Original data distribution:", dict(original_dist))
    print("Neighbor data distribution:", dict(neighbor_dist))
    print(f"exp(epsilon): {math.exp(epsilon):.4f}\n")
    
    all_keys = set(original_dist.keys()).union(set(neighbor_dist.keys()))
    warnings = False
    for k in all_keys:
        p_orig = original_dist.get(k, 0.0)
        p_neigh = neighbor_dist.get(k, 0.0)
        if p_neigh == 0 and p_orig > 0:
            print(f"[Warning] For count={k}, p_orig > 0 but p_neigh = 0, which exceeds exp(epsilon) bound.")
            warnings = True
            continue
        if p_neigh > 0:
            ratio = p_orig / p_neigh
            if ratio > math.exp(epsilon) or ratio < 1.0 / math.exp(epsilon):
                print(f"[Warning] For count={k}, ratio={ratio:.4f} exceeds exp(epsilon) bound.")
                warnings = True
    if not warnings:
        print("All probability ratios are within the exp(epsilon) bound.")

def basic_composition(epsilons):
    return sum(epsilons)

def pufferfish_check(original_data, dp_function, epsilon, 
                     target_index=0, secret1='Flu', secret2='Cancer', trials=100):
    """
    Pufferfish-style check:
     - Compare outputs when target record's attribute is secret1 vs secret2.
     - We measure, for example, how many 'Flu' appear in each sanitized outcome.
     - Then check if the ratio is bounded by exp(epsilon).
    """
    data_secret1 = copy.deepcopy(original_data)
    data_secret2 = copy.deepcopy(original_data)
    
    if 0 <= target_index < len(data_secret1):
        data_secret1[target_index]['Disease'] = secret1
    if 0 <= target_index < len(data_secret2):
        data_secret2[target_index]['Disease'] = secret2
    
    def count_flu(sanitized):
        return sum(1 for r in sanitized if r.get('Disease') == 'Flu')
    
    dist_secret1 = []
    dist_secret2 = []
    
    for _ in range(trials):
        dp_result_s1 = dp_function(data_secret1)
        dp_result_s2 = dp_function(data_secret2)
        
        dist_secret1.append(count_flu(dp_result_s1))
        dist_secret2.append(count_flu(dp_result_s2))
    
    counter_s1 = Counter(dist_secret1)
    counter_s2 = Counter(dist_secret2)
    
    # Normalize
    for k in counter_s1:
        counter_s1[k] /= trials
    for k in counter_s2:
        counter_s2[k] /= trials
    
    print(f"Pufferfish Check: record[{target_index}] Disease={secret1} vs {secret2}")
    print("Distribution of 'Flu' counts when target has secret1:", dict(counter_s1))
    print("Distribution of 'Flu' counts when target has secret2:", dict(counter_s2))
    
    all_keys = set(counter_s1.keys()).union(set(counter_s2.keys()))
    within_bound = True
    for k in all_keys:
        p_s1 = counter_s1.get(k, 0.0)
        p_s2 = counter_s2.get(k, 0.0)
        if p_s1 == 0 and p_s2 == 0:
            continue
        elif p_s1 == 0 and p_s2 > 0:
            ratio = p_s2 / (p_s1 + 1e-15)
        elif p_s2 == 0 and p_s1 > 0:
            ratio = p_s1 / (p_s2 + 1e-15)
        else:
            ratio = p_s1 / p_s2
        
        if ratio > math.exp(epsilon) or ratio < (1.0 / math.exp(epsilon)):
            print(f"[Warning] For count={k}, ratio={ratio:.4f} not within [1/e^ε, e^ε].")
            within_bound = False
    
    if within_bound:
        print("All probability ratios are within the exp(epsilon) bound (Pufferfish-style check).")
    else:
        print("Some probability ratios exceed the exp(epsilon) bound.")

def apply_pufferfish_binary(data, attribute, secret1, secret2, epsilon):
    """
    A simple mechanism to protect a binary secret (e.g., 'Flu' vs 'Cancer')
    under Pufferfish privacy by randomly flipping between secret1 and secret2 
    with probability (1 - p), where p satisfies ratio = p/(1-p) = e^epsilon.
    """
    sanitized_data = copy.deepcopy(data)
    
    # Probability p such that p/(1-p) = e^epsilon
    # => p = e^epsilon / (1 + e^epsilon)
    p = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    
    for record in sanitized_data:
        # Only flip if the attribute is actually one of the two secrets
        if attribute in record and record[attribute] in [secret1, secret2]:
            actual = record[attribute]
            # Flip with probability (1 - p)
            if np.random.rand() < p:
                # keep as is
                record[attribute] = actual
            else:
                # flip to the other secret
                record[attribute] = secret2 if actual == secret1 else secret1
    
    return sanitized_data


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

np.random.seed(42)

epsilon_age = 3.0
epsilon_disease = 3.0

sanitized_data = apply_differential_privacy(data, epsilon_age, epsilon_disease)

print("Original Data:")
for record in data:
    print(record)

print("\nSanitized Data with Differential Privacy Applied:")
for record in sanitized_data:
    print(record)

def dp_function(d):
    return apply_differential_privacy(d, epsilon_age, epsilon_disease)

dp_privacy_check(
    original_data=data,
    dp_function=dp_function,
    epsilon=epsilon_disease,
    attribute='Disease',
    trials=100
)

pufferfish_check(
    original_data=data,
    dp_function=dp_function,
    epsilon=epsilon_disease,
    target_index=0,
    secret1='Flu',
    secret2='Cancer',
    trials=100
)

eps_composed_basic = basic_composition([epsilon_age, epsilon_disease])
print(f"\nBasic Composition: epsilon_total = {eps_composed_basic}")

print("\nApplying Pufferfish Binary Flipping Mechanism")
epsilon_pufferfish = 1.0
sanitized_data_pf = apply_pufferfish_binary(data, 'Disease', 'Flu', 'Cancer', epsilon_pufferfish)
print("Original Data:")
for record in data:
    print(record)
print("\nSanitized Data (Pufferfish Binary Flipping):")
for record in sanitized_data_pf:
    print(record)
