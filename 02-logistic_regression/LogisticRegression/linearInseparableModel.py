import numpy as np
import matplotlib.pyplot as plt
import dataPretreatment
import gradientDescent
import sigmoidFunction
import featureMapping

path='E:\\Mountains\\MechaineLearning&DeepLearning\\Andrew Ng\\ML\\Coursa-homework-byHzj\\02-logistic_regression\\ex2data2.txt'
if path=='':
    print("Error path,choose you path again\n")

[score,y]=dataPretreatment.dataPretreatment(path)
rawScore=score    # a backup of raw score data

orders=6       # importannt,n is the order of high-order items,if orders not equal to 1,means it's linear inseparable, also used to control process
lamda=2
alpha=0.005    # set the learning rate alpha and iteration numbers iters
iters=30000    # set iteration number

score=featureMapping.featureMapping(score,orders)
[row,col]=score.shape

theta=np.random.randint(-10,10,col)      

[cost,theta]=gradientDescent.gradientDescent(theta,score,y,alpha,iters,orders,lamda)

# caculate the precission
predictions=sigmoidFunction.sigmoidFunction(theta,score)
predictions[predictions>=0.5]=1
predictions[predictions<0.5]=0
print('The precission of this model is: ',np.mean(predictions==y))

# plot the cost-iteration curve
fig,ax = plt.subplots()			#create a draw instance 
ax.plot(np.arange(iters),cost)
ax.set(xlabel='iters',ylabel='cost',title='Cost-Iters curve')
plt.show()

# plot the decision boundary
fig,ax = plt.subplots(figsize=(7,5))

index0=np.argwhere(y==0)   # plot raw data
index1=np.argwhere(y==1)
score1=rawScore[:,1]
score2=rawScore[:,2]
ax.scatter(score1[index0], score2[index0],c='r',marker='x',label='y=0')
ax.scatter(score1[index1], score2[index1],c='b',marker='o',label='y=1')

x = np.linspace(-1.2,1.2,200)   # the y axis correspond to the exam2
xx,yy=np.meshgrid(x,x)
temp=np.zeros((40000,3))        # the 3 colums correspond to the featureMapping function, it's wordy, column 1 is useless,it will be abandoned in featureMapping
temp[:,1]=xx.ravel()
temp[:,2]=yy.ravel()

z=featureMapping.featureMapping(temp,orders)
z=z@theta
zz=z.reshape(xx.shape)

plt.contour(xx,yy,zz,0)
plt.ylim(-1,1.2)

ax.legend()        # show labels
ax.set(xlabel='score1', ylabel='score2')
plt.show()
