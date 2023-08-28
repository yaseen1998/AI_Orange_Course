import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('csv/CarPrice_Assignment.csv') # PATH HERE
df.head()


df_encoding = pd.get_dummies(df)

# # set the independent variables and drop car_ID since it is irrelevant
X = df_encoding.drop(['car_ID', 'price'], axis=1)
# # set the dependent variable
y = df_encoding['price']

# perform 70-30 split on dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Define a LinearRegression Model without regularisation(using LinearRegression())
liner_model = LinearRegression()

# Train the model on training data(using 'model.fit()')
fit = liner_model.fit(X_train, y_train)

# Calculate Accuracy(using 'model.score()')
score = liner_model.score(X_test, y_test)

#Showcase the updated weights of the model
updated_weights = liner_model.coef_

# Define a LinearRegression Model with L1 regularisation(using Lasso())
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
lasso_score = lasso_model.score(X_test, y_test)
lasso_weights = lasso_model.coef_

# Define a LinearRegression Model with L2 regularisation(using Ridge())
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
ridge_score = ridge_model.score(X_test, y_test)
ridge_weights = ridge_model.coef_

df_heart = pd.read_csv('csv/heart.csv') # PATH HERE
X_heart = df.drop(columns = 'target', axis = 1) # set the independent variables 
y_heart = df['target'] # set dependent variable
X_train,X_test,y_train,y_test = train_test_split(X_heart,y_heart,test_size=0.3) # perform 70-30 split on data


# Define a LogisticRegression Model(using LogisticRegression())
logistic_model = LogisticRegression()
# Train the model on training data(using 'model.fit()')
logistic_model.fit(X_train, y_train)
# Gather the predictions made by the fitted model on testing data(using 'model.predict()')
model_predictions = logistic_model.predict(X_test)

# Calculate Accuracy(using 'accuracy_score()')
score = accuracy_score(y_test, model_predictions)
