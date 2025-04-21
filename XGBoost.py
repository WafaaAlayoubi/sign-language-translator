import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import xgboost as xgb
import numpy as np

# Load your CSV data
df = pd.read_csv('./dataset/datasetall.csv')
results = df['number']
features = df.drop('number', axis=1)

# Check for class imbalance
class_distribution = results.value_counts()
print("Class Distribution:")
print(class_distribution)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(results), y=results)
balanced_weights = {i: weight for i, weight in enumerate(class_weights)}
print("Class Weights:", balanced_weights)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(features, results, test_size=0.2, stratify=results, random_state=42)

# Initialize GridSearchCV with class_weight parameter
param_grid = {
    'n_estimators': [50, 100],  # Reduced options
    'max_depth': [3, 5],        # Reduced options
    'learning_rate': [0.1],     # Focused on a single value
    'subsample': [0.8],         # Fixed value
    'colsample_bytree': [0.8]   # Fixed value
}

# Use XGBoost without scale_pos_weight
model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the model with the best parameters
model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Check the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save the model to a file
joblib.dump(model, 'xgboost_model.pkl')
print("Model saved as 'xgboost_model.pkl'")