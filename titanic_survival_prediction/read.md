Features

Dataset: Titanic passenger dataset with features such as Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.
Model: The machine learning model used is a Random Forest Classifier.
Visualizations: Visualize survival distribution and confusion matrix using Seaborn and Matplotlib.
Model Evaluation: Evaluate the model's accuracy, precision, recall, F1-score, and generate a classification report.
Model Saving: The trained model is saved using joblib for future predictions.

Requirements

Make sure to install the following Python libraries before running the code:
pandas
numpy
seaborn
matplotlib
scikit-learn
joblib

You can install these libraries using pip: pip install pandas numpy seaborn matplotlib scikit-learn joblib

Clone the Repository:
Clone the repository to your local machine using the following command:

git clone https://github.com/Sharana1807/titanic-survival-prediction.git

Dataset:
Download the Titanic dataset from Kaggle Titanic Dataset. Save the dataset as tested.csv and place it in the project directory where the Python script is located.

Run the Script:
Once you have the dataset, you can run the Python script (titanic_model.py) to train and evaluate the model:
python titanic_model.py

Make Predictions:
After training the model, you can use it to make predictions on new data. The script contains an example of how to predict the survival of a new passenger.

Visualize Results
The script will display several visualizations:
Survival Distribution: A count plot showing the distribution of survival status.
Confusion Matrix: A heatmap showing the confusion matrix for model evaluation.

Save the Model
After training, the model is saved as titanic_model.pkl using the joblib library. This allows you to load and reuse the trained model without retraining it.