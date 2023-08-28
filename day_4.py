import numpy as np

def jaccard(list1, list2):
    # Convert the lists into numpy arrays 
    list1_convert = np.array(list1)
    list2_convert = np.array(list2)
    
    #Find the intersection of the two numpy arrays 
    intersection = np.intersect1d(list1_convert, list2_convert)
    
    # Find the union of the two numpy arrays 
    union = np.union1d(list1_convert, list2_convert)
    
     # Divide the length of the intersection output by the length of the union output
    divide_length = len(intersection)/len(union)
    
    return divide_length


check = None

chec2 = check or 0