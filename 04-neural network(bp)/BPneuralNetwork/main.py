import numpy as np
import thetaGroup
import matplotlib.pyplot as plt
from sigmoidFunction import sig
from scipy.optimize import minimize
from dataAndWeight import getData
from costFunction import cost
from oneHotCoder import encoder
from feedForward import FF
from gradient import gradient

path='E:\\Mountains\\MechaineLearning&DeepLearning\\Andrew Ng\\ML\\Coursa-homework-byHzj\\04-neural network(bp)\\ex4data1.mat'

np.set_printoptions(threshold=np.inf)

[rawX,rawy]=getData(path)  
# rawX 5000*401,rawy 5000*1,theta1 25*401,theta2 10*26
# theta1 is the weights between input layer and hiden layer,25 corresponds to the 25 units in hiden layer(don't include bias theta0)
# theta2 is the weights between hidden layer and output layer,26 corresponds to the 26 units in hidden layer(include theta0),10 corresponds to 10 units in output layer

y = encoder(rawy)      # change y to 5000*10
lamda=1

initial_theta=np.random.random(25*401+10*26)

final=minimize(fun=cost,
                x0=initial_theta,
                args=(rawX,y,lamda),
                method='TNC',
                jac=gradient,
                options={'maxiter':300})

final_theta=final.x

# caculate the accuracy
_,_,_,_,a3=FF(final_theta,rawX)              # a3 5000*10
predictions=np.argmax(a3,axis=1)+1           # get the max value of each row(each bit only 0 or 1), and returns the index
predictions=predictions.reshape(5000,1)

acc=np.mean(predictions==rawy)               # let the shape of predictions match rawy's

np.set_printoptions(threshold=np.inf)
# print(a3,predictions.T,'\n',rawy.T)     
print('The accurancy is ',acc)

# visualize the hidden layer
flag=0                                                          # flag to control wheather show the hidden layer,0 to skip
if flag==1:
    theta1,_=thetaGroup.thetaUnGroup(final_theta) 
    hidden_layer = theta1[:,1:]       # 25,400    

    fig,ax = plt.subplots(ncols=5,nrows=5,figsize=(8,8),sharex=True,sharey=True)    # 100 pictures in total(10*10), each one is 5×5, share x,y axis
    
    for r in range(10):               # row
        for c in range(10):           # column
            ax[5,5].imshow(hidden_layer[5 * r + c].reshape(20,20).T,cmap='gray_r')
    
    plt.xticks([])                                              # hide x,y labels
    plt.yticks([])
    plt.show()

# visualize the work process of the network
flag=1                                                          # flag to control wheather show the data,0 to skip
if flag==1:
    sampleNumber = np.random.choice(len(rawX),100)              # selected ramdomly 100 samlpes from original data
    rawimages = rawX[sampleNumber,:]                            # get out the 100 samples,100*401
    images = rawimages[:,1:]   	
    theta1,theta2=thetaGroup.thetaUnGroup(final_theta)          # extract thetas, theta1 25*401, theta2 10*26

    # plot original data figure
    fig,ax = plt.subplots(ncols=10,nrows=10,figsize=(5,5),sharex=True,sharey=True)    # 100 pictures in total(10*10), each one is 5×5, share x,y axis
    plt.xticks([])                                              # hide x,y labels
    plt.yticks([])
    for r in range(10):               # row
        for c in range(10):           # column
            ax[r,c].imshow(images[10 * r + c].reshape(20,20).T,cmap='gray_r')
        
    # plot data figure in hidden layer 
    hiddenlayer=rawimages@theta1.T                              # the 100 examples in hidden layer,100*25
    hiddenlayer=sig(hiddenlayer)
    fig,ax2=plt.subplots(ncols=10,nrows=10,figsize=(5,5),sharex=True,sharey=True)
    plt.xticks([])                                              # hide x,y labels
    plt.yticks([])
    for r in range(10):               # row
        for c in range(10):           # column
            ax2[r,c].imshow(hiddenlayer[10 * r + c].reshape(5,5).T,cmap='gray_r')

    # plot data figure in ouput layer
    hiddenlayer=np.insert(hiddenlayer,0,1,axis=1)
    outputlayer=hiddenlayer@theta2.T                            # the 100 examples in hidden layer,100*10
    outputlayer=sig(outputlayer)
    fig,ax3=plt.subplots(ncols=10,nrows=10,figsize=(5,5),sharex=True,sharey=True)
    plt.xticks([])                                              # hide x,y labels
    plt.yticks([])
    for r in range(10):               # row
        for c in range(10):           # column
            ax3[r,c].imshow(outputlayer[10 * r + c].reshape(1,10).T,cmap='gray_r')

    plt.show()