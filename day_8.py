from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import RFE


# Load Digits Data from SKlearn Library
x,y = load_digits(return_X_y=True)
print(x.shape)
print(y.shape)

# Train Perceptron model on the above dataset
model = Perceptron()
model.fit(x,y)
# Print Accuracy Previous to Dimensionality Reduction
print("Accuracy: ", model.score(x,y))

# Define a Perceptron instance to be used as classifier in Greedy Dimension Reduction
instance = Perceptron()
# Create a new RFE instance with 'n_features_to_select' set to 50.
greedy_rfe = RFE(instance, n_features_to_select=50, verbose=1)
# Fit and transform the existing data.
fit_data = greedy_rfe.fit_transform(x,y)

# Define a new Perceptron instance and train it on the transformed data.
transformed_model = Perceptron()
transformed_model.fit(fit_data, y)
# Print Accuracy after Dimensionality Reduction
print("Accuracy: ", transformed_model.score(fit_data,y))
