import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import seaborn as sns

X, Y = make_classification(
    n_samples=50,
    n_features=14,
    n_classes=4,
    n_redundant=0,
    random_state=42,
    n_clusters_per_class=5,
    n_informative=14,
)
Y = np.where(Y == 0, -1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
model = LogisticRegression(multi_class="multinomial", solver="lbfgs")
model.fit(X, Y)


print(model.predict(X), end="\n\n")
print(Y, end="\n\n")


Y_pred = model.predict(X)

# 1️⃣ **Accuracy Score**
accuracy = accuracy_score(Y, Y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 2️⃣ **Confusion Matrix**
cm = confusion_matrix(Y, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 3️⃣ **Precision, Recall, F1-score**
print("\nClassification Report:")
print(classification_report(Y, Y_pred))

# 4️⃣ **ROC Curve & AUC Score**
Y_prob = model.predict_proba(X)[:, 1]  # Get probabilities for Class 1
fpr, tpr, _ = roc_curve(Y, Y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "r--")  # Diagonal line (random model)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
