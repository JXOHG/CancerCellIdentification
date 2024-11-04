import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import tensorflow as tf

# Load the dataset
dataset = pd.read_csv('cancer.csv')
x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = dataset["diagnosis(1=m, 0=b)"]

# Split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Build the model



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(x_train, y_train, epochs=400, validation_split=0.2, callbacks=[early_stopping])



# Make predictions
predictions = model.predict(x_test)

# Binarize the predictions
predictions_binarized = (predictions > 0.5).astype(int)

# Visualization
# Plot the predictions against the actual labels
plt.figure(figsize=(12, 6))

# Scatter plot of predictions vs actual
plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('Predictions vs Actual')
plt.xlabel('Actual Diagnosis')
plt.ylabel('Predicted Diagnosis')
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.grid()

# Histogram of predictions
plt.subplot(1, 2, 2)
plt.hist(predictions, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(x_train, y_train)
print(f"Training Accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(x_test)
predictions_binarized = (predictions > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions_binarized)
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions_binarized)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, predictions_binarized)
print("Classification Report:")
print(class_report)

# ROC-AUC
roc_auc = roc_auc_score(y_test, predictions)
print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
RocCurveDisplay.from_predictions(y_test, predictions)
plt.title("ROC Curve")
plt.show()