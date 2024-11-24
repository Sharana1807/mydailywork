Credit Card Fraud Detection
Overview
This project detects fraudulent credit card transactions using machine learning. It uses the Credit Card Fraud Detection dataset to train a classification model that predicts whether a transaction is fraudulent or not.

Dataset
The dataset consists of transaction details such as:

Time: Time since the first transaction.
V1 to V28: Anonymized features.
Amount: The monetary value of the transaction.
Class: 1 for fraudulent, 0 for non-fraudulent transactions.
Libraries Used
pandas for data handling
numpy for numerical operations
scikit-learn for machine learning
imbalanced-learn for SMOTE (class balancing)
joblib for saving and loading the model
matplotlib & seaborn for visualizations

Steps to Run
Data Preprocessing:

Load the dataset, handle missing values (if any), and separate features from the target variable.
Balance the dataset using SMOTE to oversample fraudulent transactions.
Model Training:

Train a RandomForestClassifier on the data.
Evaluate the model using precision, recall, and F1-score.
Model Saving:

Save the trained model using joblib for later use.
Evaluation:

Visualize the confusion matrix to understand the model's performance.
