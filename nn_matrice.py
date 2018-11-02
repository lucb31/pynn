from tensorflow import keras
import numpy as np

# Load mnist data into array
mnist = keras.datasets.mnist
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def one_hot_encode(labels):
    result = np.zeros((len(labels), 10))
    for index, label in enumerate(labels):
        for i in range(0,10):
            if label == i:
                result[index][i] = 1
    return result

def one_hot_decode(labels):
    # Reverse encoding
    # Find max value (=1) and its index, append to array, return array
    pass

def preprocess_input(data):
    # Flatten into 1d
    input = data.flatten()
    # Normalize data
    input = np.divide(input, 255)
    return input

def calculate_outputs(input, weights):
    # Calculate Outputs neurons
    output = np.dot(weights, input)

    # Apply sigmoid function
    output = sigmoid(output)

    return output

def test(data, labels, weights):
    count_correct = 0
    count_incorrect = 0
    for i, item in enumerate(data):
        # Preprocess the data
        input = preprocess_input(item)

        # Calculate Outputs
        output = calculate_outputs(input, weights)

        result = output.argmax()
        if result == labels[i]:
            count_correct += 1
        else:
            count_incorrect += 1
    print("Accuracy rate is ", count_correct / (count_incorrect + count_correct))



# One hote encode labels
train_labels = one_hot_encode(train_labels)

# Define constants
inputs = 784
outputs = 10
epsilon = 0.01
epochs = 5

weights = np.random.random((outputs, inputs))
#print(weights.shape)

test(test_data, test_labels, weights)
for epoch in range(1,epochs + 1):
    for i, item in enumerate(train_data):
        # Preprocess the data
        input = preprocess_input(item)

        # Calculate Outputs
        output = calculate_outputs(input, weights)

        # Learning
        small_delta = train_labels[i] - output
        # Fill small_delta matrice with zeros
        # [1,2,3,4] => [ [1, 0, 0, 0], [2, 0, 0 , 0], ..... ]
        small_delta = np.hstack([small_delta.reshape((10,1)), np.zeros((10,9))])

        # Repeat input matrice in vertical direction
        # [1,2,3,4] => [[1,2,3,4], [1,2,3,4], .... ]
        input_test = np.tile(input, (10, 1))

        # calculate weight differences
        big_delta = epsilon * np.dot(small_delta, input_test)

        # apply weight differences
        weights = weights + big_delta
    print("Finished Epoch ", epoch)
    test(test_data, test_labels, weights)

test(train_data, train_labels, weights)