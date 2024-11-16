import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibrationDisplay

# Load the dataset
dataset = pd.read_csv('cancer.csv')
X = dataset.drop(columns=['diagnosis(1=m, 0=b)'])
y = dataset["diagnosis(1=m, 0=b)"]

# Create preprocessing pipeline
preprocessing = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False))
])

# Apply preprocessing
X_processed = preprocessing.fit_transform(X)
feature_names = preprocessing.named_steps['poly'].get_feature_names_out(X.columns)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Initialize both models
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # Adjust if classes are imbalanced
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Dictionary to store models
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{name} Model Evaluation:")
    print("-" * 50)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print model performance
    print(f"\n{name} Performance:")
    print(f"Training accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, test_pred_proba):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    # Check for overfitting
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    if train_accuracy - test_accuracy > 0.1:
        print(f"\nWarning: Potential overfitting detected in {name}")
        print(f"Training-Test accuracy gap: {train_accuracy - test_accuracy:.4f}")
    
    # Feature importance analysis
    if name == 'Random Forest':
        importance = model.feature_importances_
    else:  # XGBoost
        importance = model.feature_importances_
        
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot top 15 important features
    plt.figure(figsize=(10, 6))
    plt.bar(range(15), feature_importance['importance'][:15])
    plt.xticks(range(15), feature_importance['feature'][:15], rotation=45, ha='right')
    plt.title(f'Top 15 Feature Importances - {name}')
    plt.tight_layout()
    plt.show()
    
    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, test_pred_proba)
    plt.title(f"ROC Curve - {name}")
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, test_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()
    
    # Model calibration plot
    CalibrationDisplay.from_predictions(y_test, test_pred_proba)
    plt.title(f"Calibration Plot - {name}")
    plt.show()

# Compare models directly
plt.figure(figsize=(10, 6))
for name, model in models.items():
    RocCurveDisplay.from_predictions(
        y_test,
        model.predict_proba(X_test)[:, 1],
        name=name,
    )
plt.title("ROC Curves Comparison")
plt.show()

# Print final comparison
print("\nFinal Model Comparison:")
print("-" * 50)
for name, model in models.items():
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}:")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, test_pred_proba):.4f}")