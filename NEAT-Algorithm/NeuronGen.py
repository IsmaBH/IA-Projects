class NeuronGen:
    """
    This class will represents a single neuron gene for the NEAT algorithm
    Atributes:
        1) Int ID
        2) Double Bias
        3) Activation func
    Methods:
        1) Pending
    """
    def __init__(self,id,bias,activation):
        self.id = id
        self.bias = bias
        self.func = activation