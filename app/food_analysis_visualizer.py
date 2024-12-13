import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load Dataset
try:
    file_path = "/Users/nsenasabirli/Downloads/PatternProject 2/TrainingTaste_Edibility.csv"  # Update this to your dataset path
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise

# Check for missing values
if data.isnull().sum().any():
    print("Missing values detected. Filling with placeholder values.")
    data.fillna("Unknown", inplace=True)

# Ensure required columns exist
required_columns = ['food_name', 'flavors', 'user_flavor', 'match_score', 'edibility']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Convert match_score to numeric for calculations
data['match_score'] = pd.to_numeric(data['match_score'], errors='coerce').fillna(0)

# 2. Match Score Distribution
plt.figure(figsize=(12, 8))
plt.hist(data['match_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Match Scores', fontsize=16)
plt.xlabel('Match Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.savefig('match_score_distribution.png')
plt.show()

# 3. Match Score vs. Number of Ingredients (if ingredients column exists)
if 'ingredients' in data.columns:
    data['ingredient_count'] = data['ingredients'].apply(lambda x: len(str(x).split(',')))
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='ingredient_count', y='match_score', data=data, alpha=0.7, color='purple')
    plt.title('Match Score vs. Number of Ingredients', fontsize=16)
    plt.xlabel('Number of Ingredients', fontsize=14)
    plt.ylabel('Match Score', fontsize=14)
    plt.grid(True)
    plt.savefig('match_score_vs_ingredients.png')
    plt.show()
else:
    print("'ingredients' column not found. Skipping ingredient analysis.")

# 4. Clustering and Similarity Analysis
features = data[['match_score']].copy()

# Check if there are enough valid numeric values for clustering
if not features.isnull().all().all() and len(features) > 1:
    try:
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['cluster'] = kmeans.fit_predict(features)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        data['pca_one'] = pca_result[:, 0]
        data['pca_two'] = pca_result[:, 1]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='pca_one', y='pca_two', hue='cluster', data=data, palette='viridis', alpha=0.7)
        plt.title('Clustering of Dishes by Match Score', fontsize=16)
        plt.xlabel('PCA One', fontsize=14)
        plt.ylabel('PCA Two', fontsize=14)
        plt.grid(True)
        plt.legend(title='Cluster')
        plt.savefig('clustering_analysis.png')
        plt.show()
    except ValueError as e:
        print(f"Clustering failed. Error: {e}")
else:
    print("Not enough numeric values in 'match_score' for clustering.")

# 5. Topic Coherence for Predicted Flavors
flavor_counts = data['flavors'].value_counts()
plt.figure(figsize=(30, 30))
flavor_counts.head(20).plot(kind='bar', color='teal')
plt.title('Top 20 Predicted Flavors by Frequency', fontsize=16)
plt.xlabel('Predicted Flavor', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y')
plt.savefig('flavors_frequencies.png')
plt.show()

print("All analyses completed. Check saved plots for results.")
