import numpy as np
# import  Perceptron from sklearn
from sklearn.linear_model import Perceptron

class MyPerceptron(object):
    # Define the __init__ method which initialize the object’s attributes 
    # weights_bias is a numpy array. 1st element is the bias. Remaining elements are the weights.
    # iterations is the number of times the process will run to optimize the weights and bias. 
    # learning_rate is constant that controls the updates.
    def __init__(self, weights_bias, iterations=100, learning_rate=0.01):
        self.weights_bias = weights_bias
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.errors = []
        
    # Define a method for predicting the output given one input sample
    def predictOne(self, inputs):
        # B.2 Calculate the weighted sum 
        summation = np.dot(inputs, self.weights_bias[1:]) + self.weights_bias[0]
        
        # B.3 Set the activation value based on the summation result
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation
    
    # Define a method for predicting the output given a numpy array of inputs
    def predict(self, training_inputs):
        # B.4 Using predictOne method, loop through the training_inputs and predict the output 
        outputs = []
        for i in range(len(training_inputs)):
            outputs.append(self.predictOne(training_inputs[i]))
        return outputs
    
    # Define a method for training a percepton. 
    # The weights and bias are updated when the prediction is not correct. 
    # The process runs based on the defined number of iterations. 
    def train(self, training_inputs, labels):
        # B.4 Loop through the iterations
        for _ in range(self.iterations):
            # B.5 Loop through the input and label samples
            for inputs, label in zip(training_inputs, labels):
                # B.6 Predict the output using the predictOne method
                prediction = self.predictOne(inputs)
                # B.7 Update the weights and bias if the prediction is not correct
                self.weights_bias[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights_bias[0] += self.learning_rate * (label - prediction)
            # B.8 Calculate the error using the calculateError method
            error = self.calculateError(training_inputs, labels)
            # B.9 Append the error to the errors list
            self.errors.append(error)
        return self.errors
    
    # Define a method for calculating the error
    def calculateError(self, training_inputs, labels):
        # B.10 Predict the output using the predict method
        predictions = self.predict(training_inputs)
        # B.11 Calculate the error as the sum of the absolute difference between the predicted and actual labels
        error = np.sum(np.abs(predictions - labels))
        return error
        
        
training_data = np.array([[10, 143], [1, 87], [10, 135], [8, 123], [7, 125], [2, 81], [7, 118], 
                          [3, 92], [1, 85], [8, 148]])
labels = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 1])

# C.1 Create an instance of MyPerceptron with the following parameters:
# bias = 0.5, w1 = 0.5, w2 = 0.5, iterations=50, learning_rate=0.2
bias = 0.5
w1 = 0.5
w2 = 0.5
iterations = 50
learning_rate = 0.2
weights_bias = np.array([bias, w1, w2])
my_perceptron = MyPerceptron(weights_bias, iterations, learning_rate)

# C.2 Train the perceptron by passing the training_data and labels
my_perceptron.train(training_data, labels)


# C.3 Predict the outputs and calculate the accuracy of the model
predictions = my_perceptron.predict(training_data)
accuracy = np.sum(predictions == labels) / len(labels)
print("Accuracy: ", accuracy)

# D.1 Create an instance of sklearn Perceptron with the following parameters:
max_iter = 50
eta0 = 0.2
shuffle = False
random_state=1
sklearn_perceptron = Perceptron(max_iter=max_iter, eta0=eta0, shuffle=shuffle, random_state=random_state)
# D.2 Train the perceptron by passing the training_data and labels
sklearn_perceptron.fit(training_data, labels)
# D.3 Predict the outputs and calculate the accuracy of the model
predictions = sklearn_perceptron.predict(training_data)
accuracy = np.sum(predictions == labels) / len(labels)
print("Accuracy: ", accuracy)