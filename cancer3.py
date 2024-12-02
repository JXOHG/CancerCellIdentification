import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import tensorflow as tf

# Load the dataset
data = pd.read_csv("./cancer.csv")

# Prepare the features and target
X = data.drop(columns=['diagnosis(1=m, 0=b)'])
y = data['diagnosis(1=m, 0=b)']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define the functional model
input_layer = Input(shape=(X_train.shape[1],))
hidden_layer1 = Dense(64, activation='relu')(input_layer)
hidden_layer2 = Dense(32, activation='relu')(hidden_layer1)
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

# Construct the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, callbacks=[early_stopping], verbose=1)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

#Visualization of predictions vs actual values
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

# Print model evaluation metrics
print("\n=== Model Performance Summary ===")

# Training metrics
loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
print(f"\nTraining Set Performance:")
print(f"Training Accuracy: {accuracy:.4f}")
print(f"Training Loss: {loss:.4f}")

# Test set metrics
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Set Performance:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Calculate and print the difference
print(f"\nOverfitting Indicators:")
print(f"Accuracy Difference (Train-Test): {(accuracy - test_accuracy):.4f}")
print(f"Loss Difference (Test-Train): {(test_loss - loss):.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions_binarized)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, predictions_binarized)
print("\nClassification Report:")
print(class_report)

# ROC-AUC
roc_auc = roc_auc_score(y_test, predictions)
print(f"\nROC AUC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, predictions)
plt.title("ROC Curve")
plt.grid(True)
plt.show()
