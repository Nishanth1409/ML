<<<<<<< HEAD
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
n_columns = 3
n_rows = (len(df.select_dtypes(include=['float64', 'int64']).columns) * 2 + n_columns - 1) // n_columns
fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 10))
axes = axes.flatten()
columns = df.select_dtypes(include=['float64', 'int64']).columns
for i, column in enumerate(columns):
 ax = axes[i]
 df[column].hist(bins=30, edgecolor='black', ax=ax)
 ax.set(title=f"Histogram of {column}", xlabel=column, ylabel="Frequency")
 ax = axes[len(columns) + i]
 df.boxplot(column=column, grid=False, ax=ax)
 ax.set(title=f"Box Plot of {column}")
plt.tight_layout()
plt.savefig("1.png")
plt.show()
=======
import matplotlib
matplotlib.use('TkAgg') # Use TkAgg backend
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
# Load the California Housing Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
# Set up the grid layout
n_columns = 3
n_rows = (len(df.select_dtypes(include=['float64', 'int64']).columns) * 2 + n_columns - 1) // n_columns
fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 10))
axes = axes.flatten()
# Create plots
columns = df.select_dtypes(include=['float64', 'int64']).columns
for i, column in enumerate(columns):
 # Histogram
 ax = axes[i]
 df[column].hist(bins=30, edgecolor='black', ax=ax)
 ax.set(title=f"Histogram of {column}", xlabel=column, ylabel="Frequency")
 # Box Plot
 ax = axes[len(columns) + i]
 df.boxplot(column=column, grid=False, ax=ax)
 ax.set(title=f"Box Plot of {column}")
# Adjust layout and display
plt.tight_layout()
plt.savefig("1.png")
plt.show()
>>>>>>> 18dd9f31c199c0697a429e9a044efe14f494798d
