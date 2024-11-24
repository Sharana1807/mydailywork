# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib  # For saving and loading the model

# Load the dataset
file_path = 'movie_db.csv'  # Update with your dataset path
data = pd.read_csv(file_path)
print("Dataset loaded successfully!")
print("Dataset Columns:", data.columns)

# Handle missing values and ensure consistent data types
data['Duration'] = data['Duration'].str.extract(r'(\d+)').astype(float)
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')  # Convert to numeric, coerce errors to NaN
data.fillna({'Rating': data['Rating'].mean()}, inplace=True)  # Fill missing ratings with the mean

# Combine 'Actor 1', 'Actor 2', 'Actor 3' into a single feature
data['Actors'] = data['Actor 1'].fillna('') + ' ' + data['Actor 2'].fillna('') + ' ' + data['Actor 3'].fillna('')

# Convert categorical features into numerical representations
for col in ['Genre', 'Director', 'Actors']:
    data[col] = data[col].astype('category').cat.codes

# Prepare features (X) and target (y)
features = ['Genre', 'Director', 'Actors']  # Focus on the specified features
X = data[features]
y = data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the regression model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save the trained model to a file
model_filename = 'movie_rating_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Test the model with a sample input
sample_input = pd.DataFrame({
    'Genre': [2],  # Replace with actual category codes for testing
    'Director': [15],
    'Actors': [30]
})

sample_prediction = model.predict(sample_input)
print("Sample Movie Rating Prediction:", sample_prediction)

# Load the saved model (for demonstration)
loaded_model = joblib.load(model_filename)
#print(f"Model loaded from {model_filename}")

# Test the loaded model
loaded_sample_prediction = loaded_model.predict(sample_input)
print("Sample Movie Rating Prediction from loaded model:", loaded_sample_prediction)
