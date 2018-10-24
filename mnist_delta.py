from lib import Network
import mnist

# Initializing network
my_network = Network()

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Initializing our mnist network
my_network.create_mnist_network()

# Predefining labels and weights for learning
default_shoulds = []    # 1 col for every output neuron
for i in range(0,10):
    default_shoulds.append(0)

epsilon = 0.01

train_images = [train_images[0]]
train_labels = [train_labels[0]]

iteration = 0
while True:
    print ("Starting learn iteration %s:" % iteration)
    for i, image in enumerate(train_images):
        shoulds = default_shoulds
        shoulds[train_labels[i]] = 1

        # Fill shoulds with train labels
        # for index in range(0,10):
        #     shoulds.append(0)
        #     if train_labels[i] == index:
        #         shoulds[index] = 1

        # Fill inputs with train data
        my_network.read_mnist_image(image)
        my_network.delta_learning(shoulds, epsilon)
        print(i)
    print ("Finished iteration %s" % iteration)
    iteration += 1

"""
for image in test_images:
    shoulds = []
    # Fill shoulds with train data
    for label in train_labels:
        for i in range(0,10):
            if label == i:
                shoulds[i] = 1
            else
                shoulds[i] = 0
"""

print (train_labels[0])
print ("number of output neurons: %s, number of input neurons: %s" % (len(my_network.output_neurons), len(my_network.input_neurons)))
print ("Number of images: %s, number of rows: %s, number of cols: %s" % (len(train_images), len(train_images[0]), len(train_images[0][0])))

# Running delta_learning 50 times
# for i in range(0,50):
#     print_outputs(my_network, shoulds)
#     my_network.delta_learning(shoulds, epsilon)
