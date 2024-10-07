# CreditCardFraudDetection using machine learning algorithms

Overview
This project demonstrates how various machine learning algorithms can be applied to detect fraudulent credit card transactions. The dataset used is highly imbalanced, with only a small fraction of fraudulent transactions. The main challenge is to build models that can accurately classify fraudulent transactions (Class = 1) while handling the class imbalance.

Dataset
The dataset is sourced from Kaggle's Credit Card Fraud Detection dataset. It contains 284,807 transactions, of which 492 (0.17%) are fraudulent. The dataset features 30 input variables (V1-V28 are principal components from PCA, and Time and Amount are original features) along with the target variable Class.
Class 0: Legitimate Transactions
Class 1: Fraudulent Transactions

Project Workflow
1) Data Preprocessing:
Load and inspect the dataset.
Handle missing data (none in this dataset).
Explore the distribution of legitimate and fraudulent transactions.

2) Exploratory Data Analysis (EDA):
Visualize the class distribution using bar plots.
Analyze the transaction amounts and statistical differences between fraud and non-fraud transactions.

3)Data Balancing:
Due to the extreme imbalance in the dataset, an undersampling technique is applied.
Create a new dataset by combining all fraudulent transactions and a randomly selected subset of legitimate transactions.

4)Feature-Target Split:
The target variable is Class.
The input features are all other columns except Class.

5)Train-Test Split:
Split the data into training and test sets (80%-20%).

6) Model Training and Evaluation:
Train and evaluate five different machine learning models:
A) Logistic Regression
B) K-Nearest Neighbors (KNN)
C) Decision Tree
D) Random Forest
E) XGBoost
For each model:
Train the model on the training set.
Make predictions on the test set.
Evaluate the model using confusion matrices, ROC curves, and AUC scores.

Models and Performance
A) Logistic Regression
Confusion Matrix: See output in the code.
AUC Score: 0.979
ROC Curve: Plotted.

B) K-Nearest Neighbors (KNN)
Confusion Matrix: See output in the code.
AUC Score: 0.693
ROC Curve: Plotted.

C) Decision Tree
Confusion Matrix: See output in the code.
AUC Score: 0.943
ROC Curve: Plotted.

D) Random Forest
Confusion Matrix: See output in the code.
AUC Score: 0.977
ROC Curve: Plotted.

E) XGBoost
Confusion Matrix: See output in the code.
AUC Score: 0.984
ROC Curve: Plotted.

Conclusion:
XGBoost performed the best with an AUC score of 0.984, making it the most effective model for detecting fraud in this dataset.

Dependencies
To run this project, the following libraries are required:
pip install numpy pandas matplotlib scikit-learn xgboost

Python Libraries:
numpy
pandas
matplotlib
scikit-learn
xgboost

Usage:
1) Clone this repository: git clone <repo-url>
2) Install the required dependencies: pip install -r requirements.txt
3) Run the code: python credit_card_fraud_detection.py

Visualizations:
The project generates several visual outputs, including:
Bar plots showing the class distribution (normal vs fraudulent transactions).
ROC curves for each model, which visually compare their performance.


