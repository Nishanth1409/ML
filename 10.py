import matplotlib 
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive GUI rendering 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import load_breast_cancer 
from sklearn.preprocessing import StandardScaler 
# Load the breast cancer dataset 
data = load_breast_cancer() 
X = data.data 
y = data.target 
# Standardize the data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
 
# Apply KMeans clustering 
kmeans = KMeans(n_clusters=2, random_state=42) 
y_kmeans = kmeans.fit_predict(X_scaled) 
 
# Visualize the clustering result  
plt.figure(figsize=(10, 6)) 
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', edgecolors='k') 
plt.title('K-Means Clustering (2D) on Wisconsin Breast Cancer Data') 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.colorbar(label='Cluster') 
# Show the plot interactively using TkAgg 
plt.tight_layout()
plt.savefig("10.png")
plt.show() 
# Optionally, print cluster centers 
print("Cluster centers:\n", kmeans.cluster_centers_)