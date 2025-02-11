from sklearn import svm, datasets
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import numpy as np

digits = datasets.load_digits()

images = digits.images.reshape((len(digits.images), -1))
targets = digits.target


X_train, X_test, y_train, y_test = train_test_split(
    images, targets, test_size=0.2, shuffle=False, random_state=42
)

model = svm.SVC()
model.fit(X_train, y_train)

predicted = model.predict(X_test)

Q = {
    "ccuracy": accuracy_score(y_test, predicted),
    "R2": r2_score(y_test, predicted),
    "MAE": mean_absolute_error(y_test, predicted),
    "MSE": mean_squared_error(y_test, predicted),
    "RMSE": np.sqrt(mean_absolute_error(y_test, predicted), dtype="float64"),
}


print(images[0], "\n")
print(Q, "\n")

confusion_matrix = confusion_matrix(y_test, predicted)
