# credit-risk-classification

## Overview of the Analysis

The purpose of this analysis was to develop a machine learning model to classify credit risks into two categories: healthy loans and high-risk loans. By analyzing the financial data of loan applicants, we aimed to predict whether a loan would be classified as "healthy" (0) or "high-risk" (1), helping the company mitigate potential financial risks.

The data included financial features such as income, debt, and loan status. The target variable was the loan status, indicating whether the loan was "healthy" or "high-risk."

The machine learning process involved several steps:

* Data preprocessing.
* Splitting the data into training and testing sets.
* Training a logistic regression model.
* Evaluating the model’s performance using metrics such as accuracy, precision, recall, and a confusion matrix.

Tools used:

* train_test_split: We used it to divide the dataset into two parts, a training set and a testing set. This allows us to train the machine learning model on one portion of the data  and then evaluate its performance on unseen data. 
* LogisticRegression: Is a classification algorithm that is ideal for binary classification. It models the probability of an outcome using a logistic function in this case, a high-risk loan.
* confusion_matrix: Was used to evaluate the performance of the classification model by showing the counts of true positives, true negatives, false positives, and false negatives. This helps us understand where the model is making correct predictions versus errors


## Results

Machine Learning Model 1: Logistic Regression
* Accuracy: 99%
* Precision (Healthy loan): 1.00
* Precision (High-risk loan): 0.84
* Recall (Healthy loan): 0.99
* Recall (High-risk loan): 0.94
* F1-score (Healthy loan): 1.00
* F1-score (High-risk loan): 0.89


## Summary

The logistic regression model performed exceptionally well, with a high overall accuracy of 99%. The model was nearly perfect at identifying healthy loans, with a precision and recall of 1.00 and 0.99, respectively. For high-risk loans, the model also performed well, achieving a precision of 0.84 and recall of 0.94. This indicates that while the model slightly misclassified some healthy loans as high-risk, it was still highly effective at identifying most high-risk loans.

## Recommendation:
I recommend using the logistic regression model for predicting credit risk. It has demonstrated high accuracy in classifying loans, especially in correctly identifying high-risk loans, which is critical for the company’s risk mitigation strategy.