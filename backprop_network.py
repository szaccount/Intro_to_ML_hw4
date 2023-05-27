"""

backprop_network.py

"""



import random

import numpy as np

from scipy.special import softmax

import math



class Network(object):



    def __init__(self, sizes):

        """The list ``sizes`` contains the number of neurons in the

        respective layers of the network.  For example, if the list

        was [2, 3, 1] then it would be a three-layer network, with the

        first layer containing 2 neurons, the second layer 3 neurons,

        and the third layer 1 neuron.  The biases and weights for the

        network are initialized randomly, using a Gaussian

        distribution with mean 0, and variance 1.  Note that the first

        layer is assumed to be an input layer, and by convention we

        won't set any biases for those neurons, since biases are only

        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x)

                        for x, y in zip(sizes[:-1], sizes[1:])]



    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,

            test_data, statistics=True):

        """Train the neural network using mini-batch stochastic

        gradient descent.  The ``training_data`` is a list of tuples

        ``(x, y)`` representing the training inputs and the desired

        outputs.  """

        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))

        n = len(training_data)

        # ADDED
        if statistics:
            epoch_options = []
            test_acc_results = []
            training_acc_results = []
            training_loss_results = []
        # ADDED

        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [

                training_data[k:k+mini_batch_size]

                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch, learning_rate)

            # ADDED
            test_accuracy = self.one_label_accuracy(test_data)
            if statistics:
                epoch_options.append(j)
                test_acc_results.append(test_accuracy)
                training_acc = self.one_hot_accuracy(training_data)
                training_acc_results.append(training_acc)
                training_loss = self.loss(training_data)
                training_loss_results.append(training_loss)
            #ADDED
            print ("Epoch {0} test accuracy: {1}".format(j, test_accuracy))
        
        if statistics:
            return epoch_options, test_acc_results, training_acc_results, training_loss_results







    def update_mini_batch(self, mini_batch, learning_rate):

        """Update the network's weights and biases by applying

        stochastic gradient descent using backpropagation to a single mini batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw

                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (learning_rate / len(mini_batch)) * nb

                       for b, nb in zip(self.biases, nabla_b)]



    def backprop(self, x, y):
        """The function receives as input a 784 dimensional 

        vector x and a one-hot vector y.

        The function should return a tuple of two lists (db, dw) 

        as described in the assignment pdf. """
        vs, zs = self.forward_pass(x)
        return self.backward_pass(vs, zs, y)

    def forward_pass(self, x):
        """
        Returns vectors vs and zs where each entry are the v and z values of the neurons
        in the layer (for layer vs gets 0 and zs gets x).
        The z values for the last layer are computed despite they are not needed.
        Softmax is not considered as its part of the loss function (as instructed).
        """
        vs = []
        zs = []

        updated_v = np.zeros(x.shape)
        updated_z = np.array(x)
        vs.append(updated_v)
        zs.append(updated_z)
        for layer in range(self.num_layers - 1): # don't have biases and weights for first layer
            updated_v = self.weights[layer] @ updated_z + self.biases[layer]
            vs.append(updated_v)
            updated_z = relu(updated_v)
            zs.append(updated_z)
        return vs, zs
    
    def backward_pass(self, vs, zs, y):
        """
        Implements backward pass stage of back propagation.
        Returns db, dw as required in backprop
        """
        db = [None] * (self.num_layers - 1)
        dw = [None] * (self.num_layers - 1)
        
        max_layer_indx = len(vs) - 1
        assert max_layer_indx - 1 == len(dw) - 1, "Something is wrong with the indexes"
        delta = self.loss_derivative_wr_output_activations(vs[-1], y)
        for layer in range(max_layer_indx, 0, -1):
            # index for vs,zs is adapted to index - 1 for dw (vs,zs are shifted)

            if layer == max_layer_indx: # last layer has the identity as its activation function
                common = delta
            else:
                common = np.multiply(delta, relu_derivative(vs[layer]))

            dw[layer - 1] = common @ (zs[layer - 1]).T
            db[layer - 1] = common
            delta = (self.weights[layer - 1]).T @ common

        return db, dw


    def one_label_accuracy(self, data):

        """Return accuracy of network on data with numeric labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), y)

         for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results)/float(len(data))



    def one_hot_accuracy(self,data):

        """Return accuracy of network on data with one-hot labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y))

                          for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))





    def network_output_before_softmax(self, x):

        """Return the output of the network before softmax if ``x`` is input."""

        layer = 0

        for b, w in zip(self.biases, self.weights):

            if layer == len(self.weights) - 1:

                x = np.dot(w, x) + b

            else:

                x = relu(np.dot(w, x)+b)

            layer += 1

        return x



    def loss(self, data):

        """Return the CE loss of the network on the data"""

        loss_list = []

        for (x, y) in data:

            net_output_before_softmax = self.network_output_before_softmax(x)

            net_output_after_softmax = self.output_softmax(net_output_before_softmax)

            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])

        return sum(loss_list) / float(len(data))



    def output_softmax(self, output_activations):

        """Return output after softmax given output before softmax"""

        return softmax(output_activations)



    def loss_derivative_wr_output_activations(self, output_activations, y):
        """Implements derivative of loss with respect to the output activations before softmax"""        
        return softmax(output_activations) - y





def relu(z):
    """Implements the relu function activation on each element in z."""

    return np.maximum(0,np.array(z))



def relu_derivative(z):

    """
    Implements the derivative of the relu function on each element in z.
    Negative values have derivative 0, non negative values have derivative 1.
    """
    res = np.array(z)
    res[res>=0] = 1
    res[res<0] = 0

    return res
    

