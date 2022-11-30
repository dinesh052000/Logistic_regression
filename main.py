import pandas as pd
from sklearn.model_selection import train_test_split

bankmarket = pd.read_csv(r'C:\Users\DINESH\Downloads\LogisticRegressionCreditriskPrediction1603526803544\Logistic Regression Credit risk Prediction\bank-additional-full.csv')
bankmarket.head() # Reading the first 5 rows
predictor = bankmarket[["duration","age","campaign"]]
target = bankmarket[["y"]]
train_pd,test_pd,train_tr,test_tr = train_test_split(predictor,target,test_size=0.30,random_state=42)
print("Shape of train_pd:", train_pd.shape)
print("Shape of train_tr:", train_tr.shape)
print("Shape of test_pd:", test_pd.shape)
print("Shape of test_tr:", test_tr.shape)
# Importing the required class
from sklearn.linear_model import LogisticRegression

# Creating the object of the class LogisticRegression
model = LogisticRegression()

# Fitting the model to the training data
model.fit(train_pd,train_tr)

# Getting the intercept and the coefficients of the model
print("Intercept:",model.intercept_,"\nCoefficients:", model.coef_)
print("Accuracy score of the model on training data:", model.score(train_pd, train_tr))
print("Accuracy score of the model on test data:", model.score(test_pd, test_tr))