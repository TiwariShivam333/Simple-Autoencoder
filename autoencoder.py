import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt

data=pd.read_csv('Iris.csv') #Reading data
data=data.drop('Id',axis=1) #Removing the first column that contains line number
X=data.values
X=arr[:,:-1] #Taking just the features and ignoring the labels
#X=Y[:256,:]

#Min-Max  Normalization
for i in range(len(X)):
    x=X[i]
    X[i] = (x-min(x))/(max(x)-min(x))


#Predefined hidden layer sizes for autoencoder
layer_sizes=[4,30,15,30,4]
numlayers=len(layer_sizes)-1 #Excluding Input layer
n=X.shape[0]
print(n)

#Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x.astype(float)))

#Derivative of activation function
def dersigmoid(x):
    sig=sigmoid(x)
    return sig*(1-sig)


#Initializing weights randomly for the hidden layers
W=[]
"""
for i in range(0, len(layer_sizes)-1):
    wt = np.random.randn(layer_sizes[i+1], layer_sizes[i]+1)
    W.append(wt)
"""
for (l1,l2) in zip(layer_sizes[:-1],layer_sizes[1:]):
    W.append(np.random.normal(scale=0.1,size=(l2,l1+1)))

#print(W)
#print(X.T[0])

#Forward Propagation which involves calculation of activation function at every layer
def forwardpass():
    layerinput=[]
    activation=[]
    #For every layer
    for i in range(numlayers):
        if i==0:
            x=W[0].dot(np.vstack([X.T,np.ones([1,n])]))   #x=weight*layer_input + bias
        else:
            x=W[i].dot(np.vstack([activation[-1],np.ones([1,n])]))  #x=weight*layer_input + bias

        #x=np.array(x,dtype=float)
        layerinput.append(x)
        activation.append(sigmoid(x))  #Activation calculation with the above output

        #print(len(activation[i]),len(activation[i][0]))
    return layerinput,activation



#backpropagation which involves calculation of error and updating the weight matrix based on it
def backpropagation(layerinput,activation,lr):
    delta=[]
    #Calculating error from back as final output will be compared to initial input(dataset) (Autoencoder output reflects input)
    for i in reversed(range(numlayers)):
        #For first layer comparing last layer output to real input(dataset)
        if i==numlayers-1:
            op_diff=activation[i] - X.T
            err=np.sum(op_diff**2)
            err=err/2*n
            delta.append(op_diff * dersigmoid(layerinput[i]))
            #print(len(delta),len(delta[0][0]))
        #For other layers comparing with derivative of  weight*layer_input + bias for each layer
        else:
            op_diff=W[i+1].T.dot(delta[-1])
            delta.append(op_diff[:-1,:] * dersigmoid(layerinput[i]))
        #print(W[i][:,-1]+(biasupd*lr))


        biasupd=np.sum(delta[-1],axis=1)   #Bias updation

        #Weight updation based on delta value(difference)
        if(i==0):
            W[i][:,:-1]=W[i][:,:-1]- (delta[-1].dot(X))*lr
            W[i][:,-1]=W[i][:,-1] - biasupd*lr
        else:
            W[i][:,:-1]=W[i][:,:-1]- (delta[-1].dot(activation[i-1].T))*lr
            W[i][:,-1]=W[i][:,-1] - biasupd*lr

    return err



def train():
    error=0
    lr=0.01 #Learning rate
    x1=list() #x-coordinates list
    y1=list() #y-coordinates list
    for iter in range(20):
        l,a=forwardpass()

        err=backpropagation(l,a,lr)
        error=error+err
        x1.append(iter+1)
        y1.append(err)
        print('Iteration: '+str(iter+1)+'  Error: '+str(err/(2*n)))

    plt.plot(x1,y1)
    plt.xlabel("iteration")
    plt.ylabel("Error")
    plt.show()
    print(error/1000)
    l,a=forwardpass()
    xx=a[-1].T

    #print(a[-1])
    print('\n\n')
    #if want to plot the ouptut

    """plt.figure(figsize=(20,4))
    for i in range(10):
        ax=plt.subplot(2,10,i+1)
        plt.imshow(X[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(xx[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()"""



    #Printing the input to autoencoder and output of autoencoder after training showing how close they are
    for i in range(len(X)):
        print('{0}\t {1} \n'.format(X[i], xx[i]))
    #print(X)
    #print(a[-1].T)

train()

"""
xx=a[-1].T
k=2
sparse_encoded=np.zeros((len(xx),len(xx[0])))
#print(sparse_encoded)
for i in range(len(xx)):
    ind = np.argpartition(xx[i], -k)[-k:]
    for j in ind:
        sparse_encoded[i][j]=xx[i][j]

a[-1]=list(map(list, zip(*sparse_encoded)))
#print(sparse_encoded)
    for i in range(numlayers):
        if i==0:
            x=np.vstack([X.T,np.ones([1,150])])
        else:
            x=np.vstack([activation[i-1],np.ones([1,activation[i-1].shape[1]])])

        deltaW=np.sum(x[None,:,:].transpose(2,0,1) * delta[numlayers-i-1][None,:,:].transpose(2,1,0),axis=0)
        W[i]=W[i]-lr*deltaW
        """
