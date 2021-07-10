import numpy as np
from dataAndWeight import getData
from sigmoidFunction import sig

path1='E:\\Mountains\\MechaineLearning&DeepLearning\\Andrew Ng\\ML\\Coursa-homework-byHzj\\03-neural network\\ex3data1.mat'
path2='E:\\Mountains\\MechaineLearning&DeepLearning\\Andrew Ng\\ML\\Coursa-homework-byHzj\\03-neural network\\ex3weights.mat'     # just change it

[rawX,y,theta1,theta2]=getData(path1,path2)  
# rawX 5000*401,rawy 5000*1,theta1 25*401,theta2 10*26
# theta1 is the weights between input layer and hiden layer,25 corresponds to the 25 units in hiden layer(don't include bias theta0)
# theta2 is the weights between hiden layer and output layer,26 corresponds to the 26 units in hiden layer(include theta0),10 corresponds to 10 units in output layer

                #######  input layer  #######
a1=rawX                        # a1 5000*401, the input value of input layer(also the output value)

                #######  hiden layer  #######
z2=a1@theta1.T                 # z2 5000*25, the input value of hiden layer

a2=sig(z2) 
a2=np.insert(a2,0,1,axis=1)     # a2 5000*26, the output value of hiden layer,need insert bias theta0=1
#a2=sig(z2) 

                #######  output layer  #######
z3=a2@theta2.T                 # z3 5000*10, the input value of output layer

a3=sig(z3)                     # a3 5000*10, the output value of output layer

# caculate the accuracy
predictions=np.argmax(a3,axis=1)+1        # get the max value of each row, and returns the index
predictions=predictions.reshape(5000,1)   # let the shape of predictions match y's
acc=np.mean(predictions==y)

np.set_printoptions(threshold=np.inf)
print(predictions.shape,'\n',y.shape)
print('The accurancy is ',acc)