
import numpy as np
import mnist_loader
import random

#Create a new network
class Network:
    def __init__(self,neurons):
        self.layers = len(neurons)
        self.neurons = neurons
        self.weights = []
        self.biases = []
        
        for i in range(1,self.layers):
            rows = neurons[i]
            cols = neurons[i-1]
            
            #Creates a weight numpy array with dimension = dims
            #Initializes them with random normal distributed value between -1 - +1
            
            #Zero initialization
            #layer_weight = np.zeros((rows,cols))
            layer_bias = np.zeros((rows,1))

            #Random initialization
            layer_weight = np.random.randn(rows,cols)*np.sqrt(2.0/(rows+cols))
            #layer_bias = np.random.randn(rows,1)

            self.weights.append(layer_weight)
            self.biases.append(layer_bias)

#Sigmoid function
def sigmoid(z):
    return (1.0/(1.0+np.exp(-z)))

#Activations and weighted sums (feedforward)
def feedforward(network,X):
    layer_activation_list = []
    weighted_sum_list = []
    for i in range(1,network.layers):
        if i==1:
            a = X
        else:
            a = layer_activation_list[-1]
            
        weight = network.weights[i-1]
        biases = network.biases[i-1]
        z = np.dot(weight,a) + biases
        weighted_sum_list.append(z)
        a = sigmoid(z)
        layer_activation_list.append(a)
        
    return (layer_activation_list,weighted_sum_list)

#testing network
def test(network,data):
    
    activation_list,_ = feedforward(network,data[0])
    #print "Activation_List[-1] : ",activation_list[-1]
    prediction = np.argmax(activation_list[-1],axis=0)
    
    accuracy = (sum((prediction == data[1]).astype(np.float32))/data[1].shape[0])*100
    
    return accuracy

#gradient of cost function
def cost_gradient(activation,target):

    #Quadratic cost gradient
    #return (activation - target)

    #Cross Entropy cost gradient
    return np.array(np.nan_to_num(-(target/activation)+(1-target)/(1-activation)))

#derivative of sigmoid function
def sigmoid_derivative(z):

    #print "Sigmoid_Derivatives =",sigmoid(z)*(1-sigmoid(z))
    return sigmoid(z)*(1-sigmoid(z))

#update weights of all layers
def update_weights(network,activation_list,delta_list,alpha,X):
    
    for i in range(network.layers-1):
        #print "Delta List [",i,"]:",delta_list[i]
        if i==0:
            dcdw = np.dot(delta_list[i],X.transpose())
        else:        
            dcdw = np.dot(delta_list[i],activation_list[i-1].transpose())
        dcdb = np.average(delta_list[i],axis=1).reshape(delta_list[i].shape[0],1)
        
        #print dcdw
        #print "**** Max_Before : {0} , Min_Before : {1} ".format(np.amax(network.weights[i]),np.amin(network.weights[i]))
        
        network.weights[i] -= alpha*dcdw
        network.biases[i] -= alpha*dcdb
        
        #print "Weights of {0} ".format(i),network.weights[i]
        #print "**** Max_After : {0} , Min_After : {1} ".format(np.amax(network.weights[i]),np.amin(network.weights[i]))
    return

#Backpropogation of error
def errorback(network,last_delta,weighted_list):
    
    delta_list = []
    delta_list.append(last_delta)
    
    for i in range(network.layers-2,0,-1):
        delta = (np.dot((network.weights[i]).transpose(),delta_list[-1]))*(sigmoid(weighted_list[i-1]))
        delta_list.append(delta)
        
    return delta_list

#Stochastic GD
def gradient_descent(network,train_data,valid_data,alpha,epochs,batch_size):
    
    X = train_data[0]
    target = train_data[1]

    #Testing for overfitting
    X = X[:1000]
    target = target[:1000]

    alpha = alpha/float(batch_size)

    for e in range(epochs):

        #Cannot extract coloums from numpy array so take transpose
        X_trans = X.transpose()
        Y_trans = target.transpose()

        #Extract rows (previously coloums) from input and target and pack them in a list
        train_list = [[X_trans[i],Y_trans[i]] for i in range(0,X.shape[1])]

        #Random batch_size dataset for every epoch
        random.shuffle(train_list)

        for b in range(0,len(train_list),batch_size):

            train_batch = train_list[b:b+batch_size]

            #Unpack the list and store input and target as different arrays
            Xbatch = np.array([t[0] for t in train_batch]).transpose()
            Ybatch = np.array([t[1] for t in train_batch]).transpose() 

            #Generate activations (a) and weighted list (z) for every layer and neuron 
            activation_list, weighted_list = feedforward(network,Xbatch)

            #Calculate error at the final layer
            last_layer_delta = cost_gradient(activation_list[-1],Ybatch)*sigmoid_derivative(weighted_list[-1])

            #Back propogate the error to all remaining layers
            final_delta_list = errorback(network,last_layer_delta,weighted_list)

            #Reverse a list
            final_delta_list = final_delta_list[::-1]

            #Update the old weights
            update_weights(network,activation_list,final_delta_list,alpha,Xbatch)
        
        accuracy = test(network,valid_data)
        print "Accuracy after Epoch {0}: {1}".format((e+1),accuracy)
        
    return

#temp = test_data[1]
#print (sum(temp == 0).astype(np.float32))


#Load training, validation & testing data from MNIST dataset
train_data, valid_data, test_data = mnist_loader.load_data_wrapper()

my_net = Network([784, 30, 10])

#hyper parameters
epochs = 30
alpha = 0.5
batch_size = 10

gradient_descent(my_net,train_data,valid_data,alpha,epochs,batch_size)

