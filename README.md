# Predict-Future-Sales-Challenge

Given the kaggle dataset, the task in hand is to try forecasting the total amount of products sold
in a set of shops. The dataset was provided by a Russian Software Firm - 1C Company. Applying machine
learning techniques, I tried to predict the sales of the product shop-wise by training our machine learning
classifiers with the data provided and test them for accuracy.

The classifiers I used for the prediciton include - Linear Regression model, Light GBM model, LSTM.

I adopted basic data cleaning methods before training the classifier. Also I tried filling out the
missing values, removing duplicate values, concatenating the data to produce meaningful
structure to process the data.

## LINEAR REGRESSION MODEL 
To understand the simple supervised approach I used a basic
classification technique – a linear regression model to train the classifier which assumes that
there is a linear relationship between the input attributes and the class.

## Light GBM MODEL 
 This is a grading boosting network that employs tree-based algorithms for
classification and learning. Light GBM is a new algorithm and has gained attention due to it’s
lightening speed execution and lower memory consumption.

## LSTM 
Long short-term memory networks are a specialized type of recurrent neural network
(RNN)—a neural network architecture generally used for the modeling sequential data which
come in handy with our scenario. These tend retain information for long periods of time,
allowing for important information learned early in the sequence to have a larger impact on
model decisions made at the end of the sequence which reflects on the better accuracy on the
model

Using sklearn library, I was able to achieve RMSE of 0.89 for linear regression model and 0.79 for light gbm model.
Employing kertos, the LSTM classfier was able to achieve an MSE of 1.3056 after 10 epochs.
