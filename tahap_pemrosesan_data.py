import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Preprocessing
df.drop("id", axis=1, inplace=True)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encode categorical features
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Normalize numerical features
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df.drop('stroke', axis=1))
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, inertia, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('visualisasi/elbow_method.png')

# Clustering with optimal K (misal K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df.drop('stroke', axis=1))

# Silhouette Score
silhouette_avg = silhouette_score(df.drop(['stroke', 'cluster'], axis=1), df['cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Visualize clusters with PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df.drop(['stroke', 'cluster'], axis=1))
df['pc1'] = principal_components[:, 0]
df['pc2'] = principal_components[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(df['pc1'], df['pc2'], c=df['cluster'], cmap='viridis')
plt.title('Cluster Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.savefig('visualisasi/visualisasi_clustering.png')

# Analyze cluster characteristics
cluster_means = df.groupby('cluster')[numerical_cols].mean()
print("\nRata-rata Fitur per Cluster:\n", cluster_means)

# Visualize feature contributions
cluster_means.plot(kind='bar', figsize=(10, 6))
plt.title('Feature Means per Cluster')
plt.ylabel('Mean Value')
plt.savefig('visualisasi/feature_means_per_cluster.png')

# Analyze stroke distribution
stroke_dist = df.groupby('cluster')['stroke'].mean()
print("\nDistribusi Stroke per Cluster:\n", stroke_dist)

plt.figure(figsize=(6, 4))
stroke_dist.plot(kind='bar')
plt.title('Stroke Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Mean Stroke')
plt.savefig('visualisasi/stroke_distribution.png')
