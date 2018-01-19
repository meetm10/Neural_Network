
# coding: utf-8

# In[1]:


import numpy as np
import mnist_loader
import scipy.special
import random


# In[2]:


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
            #Initializes them with random normal distributed value between 0-1
            
            #layer_weight = np.zeros((rows,cols))
            layer_weight = np.random.randn(rows,cols)*np.sqrt(6.0/(rows+cols))
            self.weights.append(layer_weight)
            #layer_bias = np.random.randn(rows,1)
            layer_bias = np.zeros((rows,1))
            self.biases.append(layer_bias)


# In[3]:


train_data, valid_data, test_data = mnist_loader.load_data_wrapper()


# In[4]:


def sigmoid(z):
    
    return (1.0/(1.0+np.exp(-z)))


# In[5]:


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
    


# In[6]: Checking Accuracy


def test(network,data):
    
    activation_list,_ = feedforward(network,data[0])
    #print "Activation_List[-1] : ",activation_list[-1]
    prediction = np.argmax(activation_list[-1],axis=0)
    
    accuracy = (sum((prediction == data[1]).astype(np.float32))/data[1].shape[0])*100
    
    return accuracy


# In[7]:


def cost_gradient(activation,target):
    #print "Cost Gradients =",activation-target
    
    return (activation - target)


# In[8]:


def sigmoid_derivative(z):
    #print "Sigmoid_Derivatives =",sigmoid(z)*(1-sigmoid(z))
    
    return sigmoid(z)*(1-sigmoid(z))


# In[9]:


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


# In[10]:


def errorback(network,last_delta,weighted_list):
    
    delta_list = []
    delta_list.append(last_delta)
    
    for i in range(network.layers-2,0,-1):
        delta = (np.dot((network.weights[i]).transpose(),delta_list[-1]))*(sigmoid(weighted_list[i-1]))
        delta_list.append(delta)
        
    return delta_list


# In[11]:


def gradient_descent(network,train_data,valid_data,alpha,epochs,batch_size):
    
    X = train_data[0]
    target = train_data[1]
    alpha = alpha/float(batch_size)

    for e in range(epochs):

        X_trans = X.transpose()
        Y_trans = target.transpose()

        train_list = [[X_trans[i],Y_trans[i]] for i in range(0,X.shape[1])]

        random.shuffle(train_list)

        for b in range(0,len(train_list),batch_size):

            train_batch = train_list[b:b+batch_size]

            Xbatch = np.array([t[0] for t in train_batch]).transpose()
            Ybatch = np.array([t[1] for t in train_batch]).transpose() 

            activation_list, weighted_list = feedforward(network,Xbatch)

            last_layer_delta = cost_gradient(activation_list[-1],Ybatch)*sigmoid_derivative(weighted_list[-1])

            final_delta_list = errorback(network,last_layer_delta,weighted_list)
            final_delta_list = final_delta_list[::-1]

            update_weights(network,activation_list,final_delta_list,alpha,Xbatch)
        
        accuracy = test(network,valid_data)
        print "Accuracy after Epoch [",e,"]",accuracy
        
    return


# In[12]:


my_net = Network([784, 30, 10])

#temp = test_data[1]
#print (sum(temp == 0).astype(np.float32))
#Reverse a list
#final_delta_list = final_delta_list[::-1]
#print final_delta_list
epochs = 50
alpha = 0.5
batch_size = 100
#update_weights(my_net,activation_list,final_delta_list,alpha,X)
#print my_net.weights
gradient_descent(my_net,train_data,valid_data,alpha,epochs,batch_size)

