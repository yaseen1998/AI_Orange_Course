# Import numpy and Counter from collections
import numpy as np
from collections import Counter

class MyKNN(object):
    # Define the __init__ method which initialize the object’s attributes 
    def __init__(self, k=3):
        # B.1 initialize the number of neighbors k
        self.k = k
        
    # Define a method for "training" the model
    def fit(self, X_train, y_train):
        # B.2 initialize the training input and output
        self.X_train = X_train
        self.y_train = y_train

        
     # Define a method to claculate the Euclidean Distance between two numpy arrays
    def euclidean_distance(self, v1, v2):
        # B.3 Caluclate the distance between v1 and v2
        return np.sqrt(np.sum((v1 - v2) ** 2))
    
    
    # Define a method that predicts the output for one sample
    def predictOne(self, x):
        # B.4 Calculate the distances between x and all samples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
         
        # B.5 Sort the distance and find the indices of the fist k neighbors (smallest distance)
        k_idx = np.argsort(distances)[:self.k]
        
        # B.6 Extract the labels of the nearest neighbors
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        
        # B.7 Find and return the most frequent label using Counter
        most_common = Counter(k_neighbor_labels).most_common(1)
        
        return most_common[0][0]
    
    # Define a method that predicts the output given a numpy array of inputs
    def predict(self, X):
        # B.8 Find the predicted label for each element in X
        predictions = [self.predictOne(x) for x in X]
        return np.array(predictions)
    
training_data = np.array([[10, 143], [1, 87], [10, 135], [8, 123], [7, 125], [2, 81], [7, 118], 
                          [3, 92], [1, 85], [8, 148]])
labels = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 1])

# C.1 Create an instance of MyKNN with number of neighbors 3
instance = MyKNN(3)
# C.2 Fit the model by passing the training_data and labels
trained_model = instance.fit(training_data, labels)
# C.3 Predict the outputs and calculate the accuracy of the model
accuracy = np.mean(instance.predict(training_data) == labels)
print("The accuracy of the model is: ", accuracy)


    