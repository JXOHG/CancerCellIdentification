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

# Build the Functional API model
input_layer = tf.keras.layers.Input(shape=(x_train.shape[1],))
hidden_layer1 = tf.keras.layers.Dense(256, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
hidden_layer2 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(hidden_layer1)
dropout_layer = tf.keras.layers.Dropout(0.3)(hidden_layer2)  # Dropout for regularization
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_layer)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
optimizer = tf.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Train the model with early stopping and model checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',  # Model will be saved with a .keras extension
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model and store history
history = model.fit(
    x_train, y_train,
    epochs=400,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Plot training history
plt.figure(figsize=(15, 5))

# Plot training & validation loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot training & validation accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Calculate loss and accuracy difference
loss_diff = np.array(history.history['loss']) - np.array(history.history['val_loss'])
acc_diff = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])

# Plot the differences
plt.subplot(1, 3, 3)
plt.plot(loss_diff, label='Loss Difference (Train-Val)')
plt.plot(acc_diff, label='Accuracy Difference (Train-Val)')
plt.title('Overfitting Indicators')
plt.xlabel('Epoch')
plt.ylabel('Difference')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Make predictions
predictions = model.predict(x_test)
predictions_binarized = (predictions > 0.5).astype(int)

# Visualization of predictions vs actual values
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
