import numpy as np

class MLP:
    def __init__(self):
        self.w_network = list()
        self.b_network = list()

    def initWeighBiastMatrix(self,v):
        """
        Method that initialize weight and bias layers of the MLP network
        It initialize the matrices the network with random numbers.
        Input: a vector (list) containing the number of neurons in each layer
        Example: vector = [4,2,3,1]
        Output: a dictionary containing the weight and bias matrix with random numbers
                weight matrices of size (2,4), (3,2), (1,3)
                bias matrices of size (2,1), (3,1), (1,1)
        """
        layers = len(v)
        for i in range(layers-1):
            w_layer = np.random.random((v[i+1],v[i]))
            self.w_network.append(w_layer)
        for i in range(layers-1):
            b_layer = np.random.random((v[i+1],1))
            self.b_network.append(b_layer)


#Test section
print("MLP 0.0.1")
v1 = [int (x) for x in input().split()]
network = MLP()
network.initWeighBiastMatrix(v1)
print(network.w_network)