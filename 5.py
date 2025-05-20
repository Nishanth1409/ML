import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
np.random.seed(42)
x_values = np.random.rand(100, 1)
y_labels = np.array(['Class1' if x <= 0.5 else 'Class2' for x in x_values.flatten()])
X_train = x_values[:50]
y_train = y_labels[:50]
X_test = x_values[50:]
y_test = y_labels[50:]
k_values = [1, 2, 3, 4, 5, 20, 30]
plt.figure(figsize=(12, 8))
for i, k in enumerate(k_values, 1):
 knn = KNeighborsClassifier(n_neighbors=k)
 knn.fit(X_train, y_train)
 y_pred = knn.predict(X_test)
 plt.subplot(3, 3, i)
 plt.scatter(X_test, y_test, color='blue', label='True Label')
 plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted Label')
 plt.title(f"KNN with k={k}")
 plt.xlabel("X value")
 plt.ylabel("Class Label")
 plt.legend(loc='best')
 plt.grid(True)
plt.tight_layout()
plt.savefig("5.png")
plt.show()
for k in k_values:
 knn = KNeighborsClassifier(n_neighbors=k)
 knn.fit(X_train, y_train)
 accuracy = knn.score(X_test, y_test)
 print(f"Accuracy for k={k}: {accuracy:.2f}")