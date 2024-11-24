# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Step 1: Load the dataset from a CSV file
file_path = 'Advertising.csv'  # Update this with the correct path to the dataset
data = pd.read_csv(file_path)

# Step 2: Inspect the dataset
print("Dataset loaded successfully!")
print(data.head())  # Display the first few rows to inspect the dataset

# Step 3: Check for missing values
print("Missing values:", data.isnull().sum())

# Step 4: Display the columns in the dataset
print("Dataset Columns:", data.columns)

# Step 5: Calculate and plot the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Step 6: Split the dataset into features (X) and target (y)
X = data[['TV', 'Radio', 'Newspaper']]  # Features (advertising spend in TV, Radio, Newspaper)
y = data['Sales']  # Target variable (sales)

# Step 7: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Standardize the features (using StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Build and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 10: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 11: Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Step 12: Visualize the actual vs predicted sales
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# Step 13: Save the trained model using pickle for future use
with open('sales_prediction_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    pickle.dump(scaler, model_file)

print("Model and scaler saved successfully as 'sales_prediction_model.pkl'")

# Step 14: Example of using the saved model for prediction
# Load the saved model and scaler
with open('sales_prediction_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
    loaded_scaler = pickle.load(model_file)

# Example of predicting sales for new data
new_data = np.array([[230.1, 37.8, 69.2]])  # Example advertising spend in TV, Radio, Newspaper
new_data_df = pd.DataFrame(new_data, columns=['TV', 'Radio', 'Newspaper'])

# Scale the new data using the loaded scaler
new_data_scaled = loaded_scaler.transform(new_data_df)

# Use the trained model to predict sales
prediction = loaded_model.predict(new_data_scaled)
print(f"Predicted Sales: {prediction[0]}")
