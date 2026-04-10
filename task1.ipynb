# PROJECT: Decision Tree Implementation
# TASK 1: Build and Visualize a Classification Model
# DELIVERABLE: Python script using Scikit-Learn

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Decision Tree
# Using 'entropy' to measure the quality of split
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 4. Model Visualization (Graphical)
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=target_names)
plt.title("Decision Tree Visualization - Iris Dataset")
plt.show()

# 5. Model Analysis (Text-based Rules)
print("--- Decision Tree Rules ---")
tree_rules = export_text(clf, feature_names=list(feature_names))
print(tree_rules)

# 6. Prediction Example
sample = [[5.1, 3.5, 1.4, 0.2]] # Example flower measurements
prediction = clf.predict(sample)
print(f"\nPrediction for {sample}: {target_names[prediction][0]}")
