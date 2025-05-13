
import numpy as np 
from scipy.io import loadmat 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
 
# Load the olivettifaces.mat file (ensure it's in the same directory or update the path) 
data = loadmat('olivettifaces.mat') 
 
# Inspect the keys in the dataset 
print("Keys in the dataset:", data.keys()) 
 
# Use 'faces' as the feature matrix 
X = data['faces']  # Features (faces), this is the matrix of images 
 
# Assuming labels are the index of faces (0-40 for 40 individuals, 10 images per individual) 
y = np.repeat(np.arange(40), 10)  # 40 classes (individuals), 10 images per class 
 
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.3, random_state=42)  # Transpose for correct shape 
 
# Create and train the Naive Bayes classifier 
model = GaussianNB() 
model.fit(X_train, y_train) 
 
# Make predictions 
y_pred = model.predict(X_test) 
 
# Calculate accuracy 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy:.4f}") 