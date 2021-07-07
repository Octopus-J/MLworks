
# -*- coding: utf-8 -*- #
# Copyright (c) ZJ Huang#
''' these codes are based on the Mechine Learning course of Andrew Ng (Coursa)'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import dataPretreatment
import gradientDescent

path='E:\\Mountains\\MechaineLearning&DeepLearning\\Andrew Ng\\ML\\Coursa-homework-byHzj\\01-linear regression\\ex1data1.txt'
if path=='':
    print("there need a path to get data")
trainingSet=dataPretreatment.dataPretreatment(path)  #get 3 sets


x=trainingSet[:,0:-1]   # x is the population and the bias which combines with the theta0
y=trainingSet[:,-1]    # y is the profit

[row,col]=x.shape
theta=np.zeros(col)

theta=np.random.randint(-10,10,size=[1,col])
#theta=[[random.randint(-10, 10) for j in range(0, col)] for i in range(0, cow)]
#theta=np.array(theta)                                                              # a brief method to create theta,called list comprehension

# for i in range(0,cow):
#     for j in range(0,col):
#         theta[i,j]=random.randint(-10,10)                                         # normal method,lengthy and wordy
# else:
#     pass

alpha=0.01          # learning rate is 0.02 
iters=1000          # the number of iterations is 1000

[theta,cost]=gradientDescent.gradientDescent(theta,x,y,alpha,iters)

print(theta,theta.shape)
#########  plot the cost-iterations curve,this part of codes learn from Fun',a csdn user,thank you!
fig,ax = plt.subplots()			#create a draw instance 
ax.plot(np.arange(iters),cost)
ax.set(xlabel='iters',ylabel='cost',title='Cost-Iters curve')
plt.show()

####   plot the fit line in the raw data
fig,ax = plt.subplots()

xx= np.linspace(y.min(),y.max(),100)    #plot the fit line
y_ = theta[0,0] + theta[0,1] * xx
ax.plot(xx,y_,'r',label='predict')

ax.scatter(x[:,1],y,label='training data')       # plot raw data scatter fig

ax.legend()         # show labels

ax.set(xlabel='populaiton',ylabel='profit')   # set names of axis
plt.show()
