import numpy as np
import data
import getFinalTheta
import sigmoidFunction

path='E:\\Mountains\\MechaineLearning&DeepLearning\\Andrew Ng\\ML\\Coursa-homework-byHzj\\03-neural network\\ex3data1.mat'

[rawX,y,data]=data.getData(path)

X=np.insert(rawX,0,1,axis=1)     # insert the bias term into rawX, x=5000*401

k=10                             # the number of labels,(0-9)
lamda=1                          # the regularization parameter

theta=getFinalTheta.finalTheta(X,y,lamda,k)           # theta,10*401

# caculate the accuracy

allPredictions=sigmoidFunction.sig(X@(theta.T))       # predictions is 5000*10,each row has 10 elements,correspond to predictions of 10 models,separatly.

predictions=np.argmax(allPredictions,axis=1)          # get the max predictions location of each row,maxOne 1*5000

print(predictions,predictions.shape)

predictions+=1                                        # to match the values of y in the .mat file


acc=np.mean(predictions==y)
print('acc=',acc)