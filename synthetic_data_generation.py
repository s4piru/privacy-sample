import pandas as pd
from ctgan import CTGAN
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelBinarizer

# Privacy Evaluation
# Function to compute uniqueness ratio
def compute_uniqueness_ratio(original_df, synthetic_df):
    original_unique = original_df.drop_duplicates()
    synthetic_unique = synthetic_df.drop_duplicates()
    ratio = len(synthetic_unique) / len(original_unique)
    return ratio

# Function to compute Jaccard Similarity for categorical columns
def compute_jaccard_similarity(original_df, synthetic_df, columns):
    similarities = {}
    for col in columns:
        lb = LabelBinarizer()
        original_bin = lb.fit_transform(original_df[col])
        synthetic_bin = lb.transform(
            synthetic_df[col].where(synthetic_df[col].isin(lb.classes_), 'Other')
        )
        jaccard_scores = []
        for i in range(original_bin.shape[1]):
            # zero_division=1 for categories that are not in the original data or categories are not in the synthetic data.
            jaccard = jaccard_score(
                original_bin[:, i], synthetic_bin[:, i], zero_division=1
            )
            jaccard_scores.append(jaccard)
        similarities[col] = jaccard_scores
    return similarities

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

df_original = pd.DataFrame(data)
print("Original Data:")
print(df_original)

# Define categorical columns (non-numeircal)
categorical_columns = ['Occupation', 'ZIP', 'Disease']

# Define CTGAN (Conditional Tabular GAN) model
model = CTGAN(epochs=100, batch_size=10, verbose=True)

# learning model
model.fit(df_original, discrete_columns=categorical_columns)

# Generate synthetic data (same number of rows as original)
synthetic_data = model.sample(len(df_original))

# Sort synthetic_data by column order of original data
synthetic_data = synthetic_data[df_original.columns]
print("\nSynthetic Data:")
print(synthetic_data)

# Calculate uniqueness ratio
uniqueness_ratio = compute_uniqueness_ratio(df_original, synthetic_data)
print(f"\nUniqueness Ratio: {uniqueness_ratio:.2f}")

# Calculate Jaccard Similarity for categorical columns
categorical_cols = ['Occupation', 'ZIP', 'Disease']
jaccard_similarities = compute_jaccard_similarity(df_original, synthetic_data, categorical_cols)
print("\nJaccard Similarity for Categorical Columns:")
for col, scores in jaccard_similarities.items():
    avg_jaccard = sum(scores) / len(scores)
    print(f"{col}: {avg_jaccard:.2f}")

print("\nStatistical Summary of Original Data:")
print(df_original.describe(include='all'))

print("\nStatistical Summary of Synthetic Data:")
print(synthetic_data.describe(include='all'))
