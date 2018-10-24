import random
import cProfile

class Connection(object):
    def __init__(self, neuron, weight):
        self.neuron = neuron
        self.weight = weight

    def add_weight(self, weight):
        self.weight += weight
    # def activate(self):
    #     # manipulate input here
    #     self.value = self.input * 1
    #
    #     return self.value
    if __name__ == "__main__":
        pass

class Neuron(object):
    """
    Neuron class consisting of two core params:
        input: Network input value
        value: Activation Level: Network input value after activation
    """
    value = 0
    def __init__(self):
        self.input = 0

    def get_value(self):
        # manipulate input here, activation function
        self.value = self.input * 1
        return self.value

    if __name__ == "__main__":
        pass

class InputNeuron(Neuron):
    """
    InputNeuron class with predefined input value
    """
    def __init__(self, input):
        self.input = input
    if __name__ == "__main__":
        pass

class WorkingNeuron(Neuron):
    """
    WorkingNeuron class, working for output or hidden neurons
        connections: List of connected neurons with their weights
    """
    def __init__(self):
        self.connections = []

    def get_value(self):
        self.value = 0
        for con in self.connections:
            self.value += con.neuron.get_value() * con.weight
        return self.value

    def get_derivated_value(self):
        # for now only identity activation function, implementation later
        return 1

    def create_connection(self, neuron, weight):
        con = Connection(neuron, weight)
        self.connections.append(con)

    def delta_learning(self, small_delta, epsilon):
        for con in self.connections:
            big_delta = epsilon * small_delta * con.neuron.get_value()
            con.add_weight(big_delta)
            # print(big_delta) #DEBUG
    if __name__ == "__main__":
        pass


class Network(object):
    def __init__(self):
        self.input_neurons = []
        self.output_neurons = []
        print ("Initializing new network")

    def __repr__(self):
        print ("hello i am a network")
        for inputs in self.input_neurons:
            print ("I, ")
        for outputs in self.output_neurons:
            print ("O,")

    def create_input_neuron(self, value):
        inputn = InputNeuron(value)
        self.input_neurons.append(inputn)
        return inputn

    def create_output_neuron(self):
        outputn = WorkingNeuron()
        self.output_neurons.append(outputn)
        return outputn

    def create_full_mesh(self, weights = [[]]):
        #randomize weights if none provided
        if len(weights[0]) == 0:
            weights = []
            print ("No weights provided, i will randomize them!")
            for o in self.output_neurons:
                row = []
                for i in self.input_neurons:
                    row.append(float(random.randint(0, 100) / 100))
                weights.append(row)

        for o, output_neuron in enumerate(self.output_neurons):
            for i, input_neuron in enumerate(self.input_neurons):
                output_neuron.create_connection(input_neuron, weights[o][i])

    def delta_learning(self, shoulds = [], epsilon = 0.01):
        if len(shoulds) < len(self.output_neurons):
            print ("not enough data provided")
            return
        for o, output_neuron in enumerate(self.output_neurons):
            should = shoulds[o]
            small_delta = output_neuron.get_derivated_value() * (should - output_neuron.get_value())
            # cProfile.runctx('output_neuron.delta_learning(small_delta, epsilon)', globals(), locals())
            output_neuron.delta_learning(small_delta, epsilon)

    def create_mnist_network(self):
        """Initializing a 28x28 input neuron matrix with 10 output neurons"""
        for row in range(0, 28):
            for pixel in range(0, 28):
                self.create_input_neuron(0)

        for i in range(0,10):
            self.create_output_neuron()
        self.create_full_mesh()

    def read_mnist_image(self, image):
        i = 0
        for row in image:
            for pixel in row:
                self.input_neurons[i].input = pixel
                i += 1

    if __name__ == "__main__":
        pass
