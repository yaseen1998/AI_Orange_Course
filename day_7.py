import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("csv/cardio.csv")#PATH HERE
df.head()


# Define "cardio" as target variable (y)
y = df["cardio"]
# Drop the "cardio" column from your dataframe and store the new dataframe in 'X'
x = df.drop("cardio", axis=1)
# Use train_test_split() to split the data into training and testing (with test_size = 0.2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a function 'get_predictions' which trains a DecisionTree model on the training data. (Use DecisionTreeClassifier())
# Return the predicted values on the test data and score of the model.
def get_predictions(x_train, y_train, x_test, y_test):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = model.score(x_test, y_test)
    print("Score: ", score)
    
    return predictions, score

# Select "age" as predictor variable
age_train = x_train[["age"]]
age_test = x_test[["age"]]
# Use the function get_predictions() to get the predicted values on the test data and score of the model.
pred = get_predictions(age_train, y_train, age_test, y_test)

# Select "smoke" as predictor variable
smoke_train = x_train[["smoke"]]
smoke_test = x_test[["smoke"]]
# Use the function get_predictions() to get the predicted values on the test data and score of the model. 
pred = get_predictions(smoke_train, y_train, smoke_test, y_test)

# Select "cholesterol", "gluc", "smoke", "alco", "active", "gender" as predictor variables
select_column =x_train[["cholesterol","gluc", "smoke", "alco", "active", "gender"]]
select_column_test = x_test[["cholesterol","gluc", "smoke", "alco", "active", "gender"]]
pred = get_predictions(select_column, y_train, select_column_test, y_test)

pred_X = get_predictions(x_train, y_train, x_test, y_test)

# Create new features and use them to develop the model.
new_df = df.copy()
new_df["bmi"] = new_df["weight"]/((new_df["height"]/100)**2)
new_df["age"] = new_df["age"]/365
new_df["age"] = new_df["age"].astype(int)
y = new_df["cardio"]
# Drop the "cardio" column from your dataframe and store the new dataframe in 'X'
x = new_df.drop("cardio", axis=1)
# Use train_test_split() to split the data into training and testing (with test_size = 0.2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Select "age" as predictor variable
age_train = x_train[["age"]]
age_test = x_test[["age"]]
# Use the function get_predictions() to get the predicted values on the test data and score of the model.
pred = get_predictions(age_train, y_train, age_test, y_test)
# Select "bmi" as predictor variable
bmi_train = x_train[["bmi"]]
bmi_test = x_test[["bmi"]]
# Use the function get_predictions() to get the predicted values on the test data and score of the model.
pred = get_predictions(bmi_train, y_train, bmi_test, y_test)
pred_X = get_predictions(x_train, y_train, x_test, y_test)