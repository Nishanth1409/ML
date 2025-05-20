import numpy as np 
from scipy.io import loadmat 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
data = loadmat('olivettifaces.mat') 
print("Keys in the dataset:", data.keys()) 
X = data['faces'] 
y = np.repeat(np.arange(40), 10)
X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.3, random_state=42)
model = GaussianNB() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy:.4f}") 