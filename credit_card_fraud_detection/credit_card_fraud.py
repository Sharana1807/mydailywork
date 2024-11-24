import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings

# Suppress warnings (including the ones for SMOTE and other deprecation warnings)
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
df = pd.read_csv('creditcard.csv')

# Step 2: Data Preprocessing
# Check for missing values in the dataset
print("Missing values in the dataset:\n", df.isnull().sum())

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Step 3: Handling Imbalanced Data using SMOTE (Synthetic Minority Oversampling Technique)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Check the distribution of classes after applying SMOTE
print("\nClass distribution after SMOTE:\n", y_res.value_counts())

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Step 5: Train the model (using RandomForestClassifier as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

# Display classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the trained model using joblib
joblib.dump(model, 'credit_card_fraud_model.pkl')
print("\nModel saved successfully.")

# Step 8: Load the saved model (for future use or prediction)
loaded_model = joblib.load('credit_card_fraud_model.pkl')
print("\nModel loaded successfully.")

# Example: Making predictions using the loaded model
predictions = loaded_model.predict(X_test)
print("\nPredictions on the test set:\n", predictions)

# Optional: If you want to save the predictions to a CSV
prediction_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
prediction_df.to_csv('credit_card_predictions.csv', index=False)
print("\nPredictions saved to 'credit_card_predictions.csv'.")

# Step 9: Visualizations

# Confusion Matrix - Heatmap
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance Plot
feature_importance = model.feature_importances_
features = X.columns

# Create a DataFrame for feature importance
feat_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

# Plotting the feature importance
plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df)
plt.title('Feature Importance')
plt.show()
