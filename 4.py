<<<<<<< HEAD
import pandas as pd
df = pd.read_csv('training_data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
def find_s_algorithm(X, y):
    hypothesis = ['?' for _ in range(X.shape[1])]
    for i in range(len(X)):
        if y[i] == 'Yes':
            for j in range(len(X.columns)):
                if hypothesis[j] == '?' or hypothesis[j] == X.iloc[i, j]:
                    hypothesis[j] = X.iloc[i, j]
                else:
                    hypothesis[j] = '?'
    return hypothesis
hypothesis = find_s_algorithm(X, y)
print("Hypothesis consistent with the positive examples:", hypothesis)
=======
import pandas as pd

# Load the dataset
df = pd.read_csv('training_data.csv')

# Assume the last column is the class (target variable)
X = df.iloc[:, :-1]  # Features (all columns except the last)
y = df.iloc[:, -1]  # Class (the last column)

# Find-S algorithm
def find_s_algorithm(X, y):
    # Initialize the hypothesis to the most general hypothesis (all attributes can be anything)
    hypothesis = ['?' for _ in range(X.shape[1])]
    # Loop through all examples in the dataset
    for i in range(len(X)):
        if y[i] == 'Yes':  # If the example is a positive example
            for j in range(len(X.columns)):
                # If the hypothesis is still general or the feature matches the example, keep it
                if hypothesis[j] == '?' or hypothesis[j] == X.iloc[i, j]:
                    hypothesis[j] = X.iloc[i, j]
                # If the feature doesn't match, make it specific to the example
                else:
                    hypothesis[j] = '?'
    return hypothesis

# Get the most specific hypothesis
hypothesis = find_s_algorithm(X, y)

# Output the hypothesis
print("Hypothesis consistent with the positive examples:", hypothesis)
>>>>>>> 18dd9f31c199c0697a429e9a044efe14f494798d
