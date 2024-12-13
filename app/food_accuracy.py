import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
file_path = '/Users/nsenasabirli/Downloads/PatternProject 2/TrainingTaste_Edibility.csv'
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise

# Ensure columns exist
required_columns = ['edibility', 'match_score']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Clean and preprocess data
# Standardize 'edibility' values
data['edibility'] = data['edibility'].str.strip().str.lower()

# Map 'edibility' to binary values (1: edible, 0: potentially spoiled)
edibility_mapping = {'edible': 1, 'potentially spoiled': 0}
data['edibility_binary'] = data['edibility'].map(edibility_mapping)

if data['edibility_binary'].isnull().any():
    raise ValueError("Unexpected values in 'edibility' column. Please check the dataset.")

# Generate a binary prediction for testing (1: correct prediction, 0: incorrect)
data['is_correct'] = data['match_score'].apply(lambda x: 1 if x >= 0.5 else 0)

# Calculate accuracy
try:
    accuracy = accuracy_score(data['edibility_binary'], data['is_correct'])
    print(f"Accuracy of the model: {accuracy:.2f}")
except ValueError as e:
    print("Error calculating accuracy:", e)
    raise

# Generate and display confusion matrix
conf_matrix = confusion_matrix(data['edibility_binary'], data['is_correct'])
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Spoiled', 'Edible'], yticklabels=['Spoiled', 'Edible'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# Classification report
class_report = classification_report(data['edibility_binary'], data['is_correct'], target_names=['Spoiled', 'Edible'])
print("Classification Report:\n", class_report)

# Visualize match_score distribution
plt.figure(figsize=(10, 6))
plt.hist(data['match_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Match Score Distribution')
plt.xlabel('Match Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('match_score_distribution.png')
plt.show()

print("Analysis completed. Check the saved plots and outputs for results.")