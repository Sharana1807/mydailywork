# Import Libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load the Dataset
data = pd.read_csv('tested.csv')  # Make sure your CSV file is named correctly
print("Dataset Loaded Successfully!")
print(data.head())

# Step 2: Explore the Data
print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nSurvival Distribution:")
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')

# Step 3: Data Cleaning
# Fill missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Drop Cabin column (too many missing values)
data.drop('Cabin', axis=1, inplace=True)

# Drop rows with missing Embarked values
data.dropna(subset=['Embarked'], inplace=True)

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Step 4: Feature Engineering
# Drop unnecessary columns
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Step 5: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Step 9: Save the Model
joblib.dump(model, 'titanic_model.pkl')
print("\nModel Saved Successfully!")

# Step 10: Load and Test the Model
loaded_model = joblib.load('titanic_model.pkl')

# Sample data for prediction (Make sure the feature names match the training data)
sample_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],  # 1 for female, 0 for male (based on your encoding)
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': [2]  # 'S' is encoded as 2
})

# Make prediction using the trained model
prediction = loaded_model.predict(sample_data)

# Output the prediction
print(f"\nSample Prediction (0=Not Survived, 1=Survived): {prediction}")

# Show all graphs at once
plt.show()  # This will display all graphs after they are created
