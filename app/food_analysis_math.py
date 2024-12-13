import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
try:
    file_path = '/Users/nsenasabirli/Downloads/PatternProject 2/TrainingTaste_Edibility.csv'
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise

# Convert match_score to numeric
data['match_score'] = pd.to_numeric(data['match_score'], errors='coerce').fillna(0)

# 1. Descriptive Statistics
desc_stats = data['match_score'].describe()
print("Descriptive Statistics for Match Score:\n", desc_stats)

plt.figure(figsize=(12, 6))
plt.bar(desc_stats.index, desc_stats.values, color='skyblue')
plt.title('Descriptive Statistics for Match Score', fontsize=16)
plt.ylabel('Value', fontsize=14)
plt.grid(axis='y')
plt.savefig('descriptive_statistics.png')
plt.show()

# 2. Correlation Analysis
if 'ingredients' in data.columns:
    data['ingredient_count'] = data['ingredients'].apply(lambda x: len(str(x).split(',')))

    plt.figure(figsize=(10, 8))
    correlation_data = data[['match_score', 'ingredient_count']].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.savefig('correlation_heatmap.png')
    plt.show()
else:
    print("'ingredients' column not found. Skipping correlation analysis.")

# 3. Variance and Standard Deviation
variance = data['match_score'].var()
std_dev = data['match_score'].std()

print(f"Variance of Match Scores: {variance}")
print(f"Standard Deviation of Match Scores: {std_dev}")

# 4. Histograms and Distribution Analysis
plt.figure(figsize=(12, 6))
sns.histplot(data['match_score'], bins=20, kde=True, color='purple')
plt.title('Match Score Distribution with KDE', fontsize=16)
plt.xlabel('Match Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.savefig('match_score_distribution_with_kde.png')
plt.show()

# 5. Boxplots and Outlier Detection
plt.figure(figsize=(12, 6))
sns.boxplot(data['match_score'], color='orange')
plt.title('Boxplot for Match Scores', fontsize=16)
plt.xlabel('Match Score', fontsize=14)
plt.grid(True)
plt.savefig('boxplot_match_scores.png')
plt.show()

# 6. Trend Analysis
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    trend_data = data.groupby(data['date'].dt.date)['match_score'].mean()

    plt.figure(figsize=(14, 6))
    plt.plot(trend_data, marker='o', linestyle='-', color='blue')
    plt.title('Trend Analysis of Average Match Score Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Average Match Score', fontsize=14)
    plt.grid(True)
    plt.savefig('trend_analysis.png')
    plt.show()
else:
    print("'date' column not found. Skipping trend analysis.")

# 7. Flavors Frequency
flavor_counts = data['flavors'].value_counts()
plt.figure(figsize=(14, 8))
flavor_counts.head(10).plot(kind='bar', color='green')
plt.title('Top 10 Flavors by Frequency', fontsize=16)
plt.xlabel('Flavors', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y')
plt.savefig('top_flavors.png')
plt.show()
