# Testing-Model-Machine-Learning
Subject: Data Mining 

This project I make to test the quality of Machine Learning model. Which one is most effective for use with Breast Cancer dataset

you can download Breast Cancer dataset by this url https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

In this example, I code with Jupyter Notebook

# Machine Learning Model
 - Logistic Regression
 - K Nearest Neighbours
 - Support Vector Machine
 - Random Forest
 - Decision Tree

# Conclusion of the test
Models prediction without any normalization or standardization
 - Logistic Regression : 0.942
 - K Nearest Neighbours : 0.901
 - Support Vector Machine : 0.860
 - Random Forest Classifier : 0.965
 - Decision Tree : 0.901

Models prediction with Normalized data
 - Logistic Regression : 0.971
 - K Nearest Neighbours : 0.971
 - Support Vector Machine : 0.982
 - Random Forest Classifier : 0.965
 - Decision Tree : 0.901

Models prediction with Standardized data
 - Logistic Regression : 0.994
 - K Nearest Neighbours : 0.977
 - Support Vector Machine : 0.994
 - Random Forest Classifier : 0.965
 - Decision Tree : 0.901

From the above accuracy scores, we can observe the following:
1. DecisionTree and RandomForestClassifier are insensitive to feature scaling.
2. Logistic Regression, KNN and SVM are sensitive to feature scaling.
3. SVM and Logistic Regression models is the highest accuracy.
