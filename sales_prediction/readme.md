Sales Prediction Using Python
This project predicts sales based on advertising spend in TV, Radio, and Newspaper using machine learning. It helps businesses forecast future sales and make informed decisions about where to invest their advertising budget.

Libraries Used
pandas: For data manipulation
numpy: For numerical operations
scikit-learn: For building the machine learning model
matplotlib & seaborn: For data visualization
Dataset
The dataset contains advertising expenditures on TV, Radio, and Newspaper, and the corresponding sales data.

Columns:

TV: Spend on TV ads (in thousands)
Radio: Spend on Radio ads (in thousands)
Newspaper: Spend on Newspaper ads (in thousands)
Sales: Sales generated (in thousands of units)

Steps
Load and clean the data: Import data and check for missing values.
Feature scaling: Standardize the data for better model performance.
Train the model: Use Linear Regression to train the model on the training data.
Make predictions: Predict sales on the test data.
Evaluate the model: Measure the model’s performance using metrics like MSE, RMSE, and R² Score.
Save the model: Store the trained model and scaler for future predictions.
Visualize results: Show actual vs predicted sales on a graph, and a correlation heatmap of the features.
Running the Code
Install the necessary libraries using pip install pandas numpy scikit-learn matplotlib seaborn.
Download the dataset (Advertising.csv) and place it in the same directory as the script.
Run the script with python sales_prediction.py.

Model Evaluation
MSE (Mean Squared Error): Measures the average of the squared differences between predicted and actual values.
RMSE (Root Mean Squared Error): A more interpretable version of MSE.
R² Score: Indicates how well the model explains the variance in the data.
Results
The code will display the performance metrics, a graph comparing actual vs predicted sales, and a heatmap showing correlations between features.

Model Saving and Loading
The trained model and scaler are saved using pickle. You can load the model later to make predictions on new data without retraining it.
