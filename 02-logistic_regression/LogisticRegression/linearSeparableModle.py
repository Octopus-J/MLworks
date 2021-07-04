import numpy as np
import matplotlib.pyplot as plt
import dataPretreatment
import gradientDescent
import sigmoidFunction

path='E:\\Mountains\\MechaineLearning&DeepLearning\\Andrew Ng\\ML\\Coursa-homework-byHzj\\02-logistic_regression\\ex2data1.txt'
if path=='':
    print("Error path,choose you path again\n")

[score,y]=dataPretreatment.dataPretreatment(path)

[cow,col]=score.shape

theta=np.random.randint(-150,150,col)      
# here is a problem that if you set the initial value of thetas too small (here means below 120),
# it can't get the right answer finally, my explaination to it is if the initial value is small, 
# then you can't get a high position in the gradient descent algorithm.(only get local optima)

alpha=0.002    # set the learning rate alpha and iteration numbers iters
iters=30000    

[cost,theta]=gradientDescent.gradientDescent(theta,score,y,alpha,iters)

# plot the cost-iteration curve
fig,ax = plt.subplots()			#create a draw instance 
ax.plot(np.arange(iters),cost)
ax.set(xlabel='iters',ylabel='cost',title='Cost-Iters curve')
plt.show()

# plot the decision boundary

para1=-theta[0]/theta[2]
para2=-theta[1]/theta[2]

fig,ax = plt.subplots(figsize=(7,5))

x = np.linspace(30,100,4)   # the y axis correspond to the exam2
y_ = para1+para2 * x
ax.plot(x,y_,color='g',label='Decision Boundary')

index0=np.argwhere(y==0)   # plot raw data
index1=np.argwhere(y==1)
score1=score[:,1]
score2=score[:,2]
ax.scatter(score1[index0], score2[index0],c='r',marker='x',label='y=0')		   
ax.scatter(score1[index1], score2[index1],c='b',marker='o',label='y=1')

ax.legend()        # show labels
ax.set(xlabel='score1', ylabel='score2')
plt.show()

# caculate the precission
predictions=sigmoidFunction.sigmoidFunction(theta,score)
predictions[predictions>=0.5]=1
predictions[predictions<0.5]=0
print('The precission of this model is: ',np.mean(predictions==y))