Dataset:
The Iris dataset is a classic dataset in the field of machine learning and is commonly used for classification tasks. It contains 150 samples from three different Iris species, with four features for each sample:
Sepal Length
Sepal Width
Petal Length
Petal Width

Requirements:
Ensure you have the following Python libraries installed:
pandas
numpy
sklearn
seaborn
matplotlib
pickle

Steps to Run the Project
Step 1: Load the Dataset
The IRIS.csv file should be placed in the same directory or the correct path should be provided in the code.
Step 2: Preprocessing
The dataset is loaded into a pandas DataFrame, and columns are renamed for easier access.
We check for missing values in the dataset to ensure that the data is clean.
Step 3: Model Training
The dataset is split into training and testing sets (80% for training, 20% for testing).
Features are standardized using StandardScaler to ensure that the machine learning model performs better.
A RandomForestClassifier is trained on the data.
Step 4: Model Evaluation
The trained model is evaluated using the test set to calculate accuracy and other metrics.
A classification report and confusion matrix are generated.
Step 5: Save the Model
The trained model and the scaler used for feature scaling are saved as .pkl files (iris_flower_classifier.pkl and scaler.pkl) using Python's pickle module.
Step 6: Prediction with Saved Model
A sample new flower's measurements are provided for prediction.
The saved model and scaler are loaded, and the new data is transformed and passed into the model to make a prediction.
