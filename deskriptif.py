import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
except FileNotFoundError:
    print("Error: healthcare-dataset-stroke-data.csv not found. Please ensure the file is in the current directory.")
    exit()

# Jumlah fitur dan observasi
num_features = len(df.columns)
num_observations = len(df)

print(f"Jumlah fitur: {num_features}")
print(f"Jumlah observasi: {num_observations}")

# Fitur relevan
relevant_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status', 'gender']

# Statistik deskriptif untuk fitur relevan
print("\nStatistik Deskriptif Fitur Relevan:")
print(df[relevant_features].describe())

# Menampilkan beberapa baris pertama dari dataset
print("\nContoh data:")
print(df.head())
