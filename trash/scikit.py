from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=2, n_classes=2, random_state=42
)


# Step 2: Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Step 3: Train a classifier (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # For AUC-ROC (probabilities)

# Step 5: Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Step 6: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# Step 7: Plot ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()
