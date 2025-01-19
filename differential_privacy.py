import copy
import numpy as np
import math
from collections import Counter

def laplace_noise(scale):
    """Generates Laplace noise with mean 0 and specified scale."""
    return np.random.laplace(0, scale)

def apply_laplace_mechanism(data, attribute, epsilon, sensitivity=1.0):
    """Applies the Laplace Mechanism to a numerical attribute in the dataset."""
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
    """Implements the Exponential Mechanism for selecting an output based on utility scores."""
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    
    # Compute the probability for each category
    exp_scores = {}
    for category, utility in utilities.items():
        exp_score = math.exp((epsilon * utility) / (2 * sensitivity))
        exp_scores[category] = exp_score
    
    total = sum(exp_scores.values())
    probabilities = [exp_scores[cat] / total for cat in utilities.keys()]
    categories = list(utilities.keys())
    
    # Select a category based on the computed probabilities
    selected = np.random.choice(categories, p=probabilities)
    
    return selected

def apply_exponential_mechanism_individual(data, attribute, epsilon, sensitivity=1.0):
    """Applies the Exponential Mechanism to a categorical attribute for each record individually."""
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    
    sanitized_data = copy.deepcopy(data)
    n = len(sanitized_data)
    
    # Allocate privacy budget per record using Basic Composition
    epsilon_per_record = epsilon / n if n > 0 else 0
    
    # Gather unique categories in the dataset
    unique_categories = list({record[attribute] for record in sanitized_data if attribute in record})
    
    for record in sanitized_data:
        if attribute in record:
            actual_category = record[attribute]
            utilities = {category: 1 if category == actual_category else 0 for category in unique_categories}
            
            # Select a new category using the Exponential Mechanism
            selected_category = exponential_mechanism_selection(utilities, epsilon_per_record, sensitivity)
            record[attribute] = selected_category
    
    return sanitized_data

def apply_differential_privacy(data, epsilon_age=1.0, epsilon_disease=1.0):
    """Applies differential privacy to the dataset by perturbing 'Age' (numeric) and 'Disease' (categorical) attributes."""
    # Apply Laplace Mechanism to 'Age'
    data_with_noisy_age = apply_laplace_mechanism(data, 'Age', epsilon_age, sensitivity=1.0)
    
    # Apply Exponential Mechanism to 'Disease'
    data_with_noisy_disease = apply_exponential_mechanism_individual(
        data_with_noisy_age, 'Disease', epsilon_disease, sensitivity=1.0
    )
    
    return data_with_noisy_disease

def dp_privacy_check(original_data, dp_function, epsilon, attribute='Disease', trials=100):
    """
    A simple check to demonstrate how Differential Privacy might limit the impact of one record
    on a query result distribution. This is not a formal proof, just an illustrative check.
    """
    # Create a neighbor dataset by removing the last record
    neighbor_data = original_data[:-1]
    
    # Measure how often a specific category appears in the sanitized data
    def count_category_in_sanitized(sanitized, category='Flu'):
        return sum(1 for r in sanitized if r.get(attribute) == category)
    
    # Repeatedly apply DP mechanism and gather the distribution
    original_counts = []
    neighbor_counts = []
    for _ in range(trials):
        dp_result_original = dp_function(original_data)
        dp_result_neighbor = dp_function(neighbor_data)
        original_counts.append(count_category_in_sanitized(dp_result_original, 'Flu'))
        neighbor_counts.append(count_category_in_sanitized(dp_result_neighbor, 'Flu'))
    
    # Convert counts to probability distribution (frequency)
    original_dist = Counter(original_counts)
    neighbor_dist = Counter(neighbor_counts)
    
    # Normalize to create probability distributions
    original_dist = {k: v / trials for k, v in original_dist.items()}
    neighbor_dist = {k: v / trials for k, v in neighbor_dist.items()}
    
    # Display results
    print(f"DP Privacy Check (Attribute: {attribute})")
    print("Distribution of 'Flu' counts in sanitized data:")
    print("Original data distribution:", dict(original_dist))
    print("Neighbor data distribution:", dict(neighbor_dist))
    print(f"exp(epsilon): {math.exp(epsilon):.4f}\n")
    
    # Calculate and report probability ratios that exceed the bound
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
    """
    Basic Composition: If you perform k mechanisms with epsilons = [ε1, ε2, ..., εk],
    then the composed mechanism is (sum of epsilons) DP.
    """
    return sum(epsilons)


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

epsilon_age = 3.0       # Privacy budget for 'Age'
epsilon_disease = 3.0   # Privacy budget for 'Disease'

sanitized_data = apply_differential_privacy(data, epsilon_age, epsilon_disease)

print("Original Data:")
for record in data:
    print(record)

print("\nSanitized Data with Differential Privacy Applied:")
for record in sanitized_data:
    print(record)

# DP function for privacy check
def dp_function(d):
    return apply_differential_privacy(d, epsilon_age, epsilon_disease)

# Perform Privacy Check
dp_privacy_check(
    original_data=data,
    dp_function=dp_function,
    epsilon=epsilon_disease,
    attribute='Disease',
    trials=100
)

# Basic Composition: The composed epsilon is the sum of individual epsilons
eps_composed_basic = basic_composition([epsilon_age, epsilon_disease])
print(f"\nBasic Composition: epsilon_total: {eps_composed_basic}")  
