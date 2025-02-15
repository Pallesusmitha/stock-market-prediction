import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Preparation
# Assuming we have a dataset 'stock_data.csv' with features and a target column
# Example features: 'Open', 'High', 'Low', 'Close', 'Volume'
# Target: 'Trend' (1 for price up, 0 for price down)

# Load dataset
df = pd.read_csv('stock_data.csv')

# Feature selection
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df[features]
y = df['Trend']  # Target variable

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Model Training using CART (Decision Tree)
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Prediction and Evaluation
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Step 5: Visualization
# Plotting the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=['Down', 'Up'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

# Plotting Feature Importance
feature_importance = clf.feature_importances_
plt.figure(figsize=(10, 5))
sns.barplot(x=features, y=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
