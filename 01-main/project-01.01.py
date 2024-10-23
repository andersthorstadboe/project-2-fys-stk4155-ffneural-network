import autograd.numpy as np
from classSupport import *

class FFNeuralNework:

    def __init__(self,
                network_input_size,
                layer_output_size
        ):
        self.net_in_size = network_input_size
        self.layer_out_size = layer_output_size

    def create_layers(self):
        """
        Creates the layers based on the initialization of the class

        """
        self.layers = []
        i_size = self.net_in_size
        for layer_out_size in self.layer_output_sizes:
            W = np.random.randn(i_size,layer_out_size)
            b = np.random.randn(layer_out_size)
            self.layers.append((W,b))

            i_size = layer_out_size

    
class LinearRegressor:

    def __init__(self):
        pass


class LogisticRegressor:

    def __init__(self):
        pass


