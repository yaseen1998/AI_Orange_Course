import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


"""Exercise 1: NumPy Arrays and Linear Algebra
Create a NumPy array based on the given problem and solve the linear system using the inv() and dot() methods. Print the solution.

4x1+3x2+2x3=25 

−2x1+2x2+3x3=−10 

3x1−5x2+2x3=−4
"""
"""
# Create a 2D numpy array(A) consisting of Equation coefficients.
# Create a 1D numpy array(B) consisting of constants
# Solve the above system of equations by inverting the 2D numpy array and then using dot product b/w inv(A) and B.
# """
A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
B = np.array([25, -10, -4])
inv_A = np.linalg.inv(A)
dot_A_B = np.dot(inv_A, B)

df = pd.read_csv("world_population.csv")
print(df.head())

# Filter the rows which has 'Jordan' as entry in the Country Column
filter_jordan = df['Country'] == 'Jordan'   
df_jordan = df[filter_jordan]
print(df_jordan.head())


# Use groupby for 'Continent' and '2022 Population'
grouby_continent = df.groupby('Continent')['2022 Population'].sum()
print(grouby_continent)

df_jordan = pd.read_csv("jordan_co_emissions.csv")

# Define x to be the year column
# Define y to be the 'Cumulative CO2 emissions' column
x = df_jordan['Year']
y = df_jordan['Cumulative CO2 emissions']

# Define an instance of 'LinearRegression()'.
linear_model = LinearRegression()

# Calculate the slope and intercept of the trained model.
# Expected 2D array, got 1D array instead:
x = np.array(x).reshape(-1, 1)
fit = linear_model.fit(x, y)
slope = fit.coef_
print("Slope: ", slope)
intercept = fit.intercept_
print("Intercept: ", intercept)
# Calculate the Coefficient of Determination(R2 score).

coef = linear_model.score(x, y)

# Showcase the values
print("Coefficient: ", coef)

# Use predict() to find the predictions of the model for x.
pred = linear_model.predict(x)
print("Predictions: ", pred)


