# train_model.py

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'iris_model.pkl')

print("Model trained and saved as 'iris_model.pkl'")
