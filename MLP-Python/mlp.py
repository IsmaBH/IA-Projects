import numpy as np

class MLP:
    """
    Class that implements a custom MLP, this class will be able to create
    a variable size MLP network with n number of neurons and it will be
    possible to use different transfer functions per layer (i.e. sigmoid, tanh)
    """
    def __init__(self):
        self.w_network = list()
        self.b_network = list()
        self.outputs = list()
        self.sensitivities = list()

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

    def activation(self, weights, bias, inputs):
        """
        Method to make the operations to activate a layer in the network
        Inputs:
            1. weights: index of the wanted layer
            2. bias: index of the wanted layer
            3. inputs: numpy array of the input(s) for the layer
        output: numpy array of the dot of the operation
        """
        a = np.dot(self.w_network[weights],inputs) + self.b_network[bias]
        return a
    
    def transfer(opc, a):
        """
        Method to select the transfer function for the given activation
        this method will increase as new transfer functions are added.
        For this version it will implement:
            1. Linear function
            2. Sigmoid function
            3. Tansig function
        Inputs:
            1. opc: number for the option selected
            2. a: numpy array of the activation matrix
        Output: numpy array with the activation matrix transformed by the desired function
        """
        if opc == 1:
            return a
        elif opc == 2:
            return 1.0/(1.0 + np.exp(-a))
        else:
            return (np.exp(a) - np.exp(-a))/(np.exp(a) + np.exp(-a))
        
    def transfer_derivative(opc,outputs,v):
        """
        Method that prepares the derivates of the implemented transfer functions
        to simplificate this process the derivates were done by hand and simplified 
        to write them in the next manner:
            1. purelin: 1
            2. sigmoid: a_n(1 - a_n)
            3. tansig: 1 - (a_n)**2
        Inputs:
            1. opc: the desired transfer function to derivate
            2. outputs: outputs of the layer
            3. v: in this case v is an int that tells how many neurons are in the purelin option
        Outputs:
            Returns the derivate matrix
        """
        if opc == 1:
            aux = np.ones(v)
            return np.diag(aux)
        elif opc == 2:
            x,y = outputs.shape
            outputs = outputs.reshape(y,x)
            aux = outputs*(1-outputs)
            return np.diag(aux)
        else:
            x,y = outputs.shape
            outputs = outputs.reshape(y,x)
            aux = 1 - np.power(outputs,2)
            return np.diag(aux)
    
    def forward_propagate(self,v,input):
        """
        Method to forward propagate the data through the network
        Inputs:
            1. v: vector containing the desired transfer function per layer
            2. input: initial data vector for the network
        Output:
            It doesn't return a special type of data but sets the network property of "outputs"
            used later for the back propagation calculations
        """
        new_inputs = []
        for i in range(len(self.w_network)):
            a = self.activation(i,i,input)
            output = self.activation(v[i],a)
            new_inputs.append(output)
            input = new_inputs[i]
        self.outputs = new_inputs

    def backpropagate_error(self,Y_target,functions,neurons):
        """
        Method that makes the function of backpropagation in the network
        Inputs:
            1. Y_target: vector (numpy array) that contains the target values
            2. functions: vector (list) of the desired transfer functions for every layer
            3. neurons: vector (list) of the desired number of neurons in each layer
        Output:
            It doesn't return a special type of databut sets the network property of "sensitivities"
            used later for the update of the parameters in the weight and bias matrices.
        """
        sens = []
        count = 0
        for i in reversed(range(len(self.outputs))):
            if i == len(self.outputs)-1:
                derivate = self.transfer_derivative(functions[i],self.outputs[i],neurons[i+1])
                error = Y_target - self.outputs[i]
                s = np.dot(((-2)*derivate),error)
                sens.append(s)
            else:
                for j in range(self.outputs):
                    derivate = self.transfer_derivative(functions[i],self.outputs[i],neurons[i+1])
                    s = np.dot(np.dot(derivate,np.transpose(self.w_network[i+1])),sens[j+count])
                    sens.append(s)
                    count += 1
                    break
        self.sensitivities = sens

    def learning_rule(self,learning_rate,a0):
        """
        Method to update the weight and bias matrices, according to the sensitivites
        obtained in the backpropagation.
        Inputs: 
            1. learning_rate: float number, also known as alfa value
            2. a0: initial vector, most of the time is the input of the network
        Output: 
            It doesn't return a special type of data but sets the network new values of
            the weight and bias matrices to use in next iterations.
        """
        new_w = []
        new_b = []
        for i in reversed(range(len(self.w_network))):
            if i == 0:
                aux1 = learning_rate*self.sensitivities[i]
                aux2 = aux1 * np.transpose(a0)
                n_w = self.w_network[i] - aux2
                n_b = self.b_network[i] - aux1
                new_w.append(n_w)
                new_b.append(n_b)
            else:
                aux1 = learning_rate*self.sensitivities[i]
                aux2 = aux1 * np.transpose(self.outputs[i-1])
                n_w = self.w_network[i] - aux2
                n_b = self.b_network[i] - aux1
                new_w.append(n_w)
                new_b.append(n_b)
        self.w_network = new_w
        self.b_network = new_b
    
    def fit(self,X_train,Y_train,l_rate=0.001,expected_error=0.0001,epoch=100,early_stop=False,val_round=0):
        """
        Method that starts the training, this method will have some default values for the trainings
        such as a learning rate of 0.001, expected error of 0.0001, epoch of 100, early stop to False
        and validation round of 0, all this parameters will be up to the user to set otherwise
        Inputs:
            1. X_train: list of features for the network to train
            2. Y_train: list of target values for the network to compare
            3. l_rate: float number for the learning rate, by default is 0.001
            4. expected_error: float number for the target error, by default is 0.0001
            5. epoch: int number for the quantity of training to do
            6. early_stop: boolean value, if true it will require a number for the validation round
            7. val_round: int number of the interval to do a validation round to check if the performance is really improving
        Output:
            It returns a trained model or an Error if the early stop was activated
        """
        pass


#Test section
print("MLP 0.0.1")
v1 = [int (x) for x in input().split()]
network = MLP()
network.initWeighBiastMatrix(v1)
print(network.w_network)