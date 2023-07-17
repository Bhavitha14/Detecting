import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Assume you have extracted features and labels from brain tumor images
features = np.array(...)  # Shape: (num_samples, num_features)
labels = np.array(...)  # Shape: (num_samples,)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a Support Vector Machine classifier
svm = SVC()

# Train the classifier
svm.fit(X_train, y_train)

# Predict on the test set
predictions = svm.predict(X_test)

# Evaluate the model
report = classification_report(y_test, predictions)
print(report)
