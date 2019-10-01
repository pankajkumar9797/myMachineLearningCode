import numpy as np
import pandas as pd


class NN:
    def __init__(self):
        self.n_units_all_layers = []
        self.activation_functions = []
        self.data_feed_index = 0

    def add_layer(self, n_units, activation_function):
        self.n_units_all_layers.append(n_units)
        self.activation_functions.append(activation_function)

    def feed_data(self, x_train, y_train, batch_size=32):
        x_data = x_train[self.data_feed_index:self.data_feed_index + batch_size, :]
        y_data = y_train[self.data_feed_index:self.data_feed_index + batch_size]
        return x_data, y_data

    @staticmethod
    def sigmoid(y_vec):
        return 1/(1 + np.exp((-1)*y_vec))

    @staticmethod
    def relu(y_vec):
        return np.maximum(y_vec, 0)

    @staticmethod
    def softmax(y_vec):
        y_vec_exp = np.exp(y_vec)
        return y_vec/np.sum(y_vec_exp)

    def initialize_weights_and_biases(self):
        """
        This function initializes the weights and the biases
        :return: a weight dict and biases dict, in weights, each key represents the matrix of
        initialized weights matrix and similary, each key in biases represents a vector of biases
        """

        """
        Initializing the weights
        """
        n_layers = len(self.n_units_all_layers)
        weights = {}
        for i in range(1, n_layers):
            weight = "W" + str(i)
            w = np.random.rand(self.n_units_all_layers[i-1], self.n_units_all_layers[i])
            weights[weight] = w

        """
        Initializing the biases
        """
        biases = {}
        if initialize_weights:
            for i in range(1, self.n_layers):
                bias = "b" + str(i)
                b = np.random.rand(self.n_units_all_layers[i-1][i], )
                biases[bias] = b

        return weights, biases

    """
    Cost is to be calculated over the batch size
    """
    @staticmethod
    def cross_entropy_cost(y_pred, y_true):
        """
        :param y_pred: A numpy array predicted labels
        :param y_true: A numpy array of actual labels of the data set
        :return: averaged cost of the data set
        """
        cost = np.multiply(y_true, np.log(y_pred)) - np.multiply((1 - y_true), np.log(1 - y_pred))
        cost = np.average(cost)
        return cost

    def feed_forward(self, x_train):
        """
        This function goes through all the layers in the neural network and calculates the output
        from the final layer
        :param x_train: batch data set
        :return: output vector from the last layer
        """

        for i in range(self.n_layers):
            y_i = self.calculate_y(x_train, i)
            if self.activation_function == "relu":
                z_i = self.relu(y_i)
                x_train = z_i
            elif self.activation_function == "sigmoid":
                z_i = self.sigmoid(y_i)
                x_train = z_i

        return x_train

    def calculate_gradients(self):
        raise NotImplementedError

    def back_propagation(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


if __name__ == '__main__':
    """
    Number of layers.
    Number of units in respective layers
    """
    nn = NN()
    nn.add_layer(5, 'relu')
    nn.add_layer(10, 'relu')
    nn.add_layer(10, 'relu')
    nn.add_layer(3, 'softmax')