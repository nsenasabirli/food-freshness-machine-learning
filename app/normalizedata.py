import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
try:
    file_path = '/Users/nsenasabirli/Downloads/PatternProject 2/TrainingTaste_Edibility.csv' # Update with your path
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise

# Check for missing values
if data.isnull().sum().any():
    print("Missing values detected. Filling with placeholder values.")
    data.fillna("Unknown", inplace=True)

# Select numeric columns for normalization
numeric_columns = ['match_score']  # Add more numeric columns if necessary
if not all(col in data.columns for col in numeric_columns):
    raise ValueError(f"Some numeric columns are missing: {numeric_columns}")

# Normalize the data using Min-Max Scaling
scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Save the normalized dataset
normalized_file_path = "/Users/rony/Downloads/PatternResourceFiles/Normalized_TrainingTaste_Edibility.csv"
data.to_csv(normalized_file_path, index=False)
print(f"Normalized dataset saved to {normalized_file_path}.")

# Plot the normalized data (for visualization)
plt.figure(figsize=(10, 6))
plt.hist(data['match_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Normalized Match Scores')
plt.xlabel('Normalized Match Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('normalized_match_score_distribution.png')
plt.show()
