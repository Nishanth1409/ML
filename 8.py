<<<<<<< HEAD
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
import numpy as np 
data = load_breast_cancer() 
X = data.data 
y = data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
clf = DecisionTreeClassifier(random_state=42) 
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy on Test Set: {accuracy:.4f}') 
new_sample = X_test[0].reshape(1, -1) 
predicted_class = clf.predict(new_sample) 
=======
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
import numpy as np 
 
# Load the Breast Cancer dataset 
data = load_breast_cancer() 
X = data.data 
y = data.target 
 
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
 
# Create a DecisionTreeClassifier instance and train it 
clf = DecisionTreeClassifier(random_state=42) 
clf.fit(X_train, y_train) 
 
# Predict the test set results 
y_pred = clf.predict(X_test) 
 
# Evaluate the classifier performance 
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy on Test Set: {accuracy:.4f}') 
 
# Classify a new sample (randomly selected from the test set for demonstration) 
new_sample = X_test[0].reshape(1, -1)  # Take the first sample from the test set 
predicted_class = clf.predict(new_sample) 
 
# Output the predicted class (0: malignant, 1: benign) 
>>>>>>> 18dd9f31c199c0697a429e9a044efe14f494798d
print(f'Predicted Class for New Sample: {"Benign" if predicted_class == 1 else "Malignant"}')