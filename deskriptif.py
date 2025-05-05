
import pandas as pd

# Load the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Display basic information about the dataset
print("Jumlah fitur:", len(df.columns))
print("Jumlah data observasi:", len(df))

# Display descriptive statistics for relevant features
relevant_features = ['age', 'avg_glucose_level', 'bmi']
print("\nStatistik deskriptif fitur yang relevan:")
print(df[relevant_features].describe())
