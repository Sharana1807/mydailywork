# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Step 1: Load the dataset from a CSV file
file_path = 'IRIS.csv'  # Update this with the correct path to the dataset
data = pd.read_csv(file_path)

# Step 2: Inspect the dataset
print("Dataset loaded successfully!")
print(data.head())  # Display the first few rows to inspect the dataset

# Step 3: Check for missing values
print("Missing values:", data.isnull().sum())

# Step 4: Display the columns in the dataset
print("Dataset Columns:", data.columns)

# Step 5: Rename columns to ensure they match the expected format
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Step 6: Split the dataset into features (X) and target (y)
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Features
y = data['species']  # Target variable (species)

# Step 7: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Standardize the features (using StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Build and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 10: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 11: Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Step 12: Save the trained model using pickle for future use
with open('iris_flower_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:  # Save the scaler for future use
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")

# Step 13: Example of using the saved model for prediction
# Load the saved model and scaler
with open('iris_flower_classifier.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Example of predicting a new sample
sample = [[7, 3.8, 1.7, 0.2]]  # Example flower measurements (sepal_length, sepal_width, petal_length, petal_width)
sample_scaled = loaded_scaler.transform(sample)  # Scale the sample using the loaded scaler
prediction = loaded_model.predict(sample_scaled)

# Display the prediction result
print(f"Predicted species: {prediction[0]}")
