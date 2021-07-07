# -*- coding: utf-8 -*- #
# Copyright (c) ZJ Huang#
''' these version is for multivariant linear regression'''

import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import dataPretreatment
import gradientDescent
from mpl_toolkits.mplot3d import Axes3D

path='E:\\Mountains\\MechaineLearning&DeepLearning\\Andrew Ng\\ML\\Coursa-homework-byHzj\\01-linear regression\\ex1data2.txt'    #type your file path
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
ax.set(xlabel='iters',ylabel='cost',title='Cost-Iter curve')
plt.show()

####   plot the fit plane in the raw data
fig = plt.figure()
ax = fig.gca(projection='3d')  # create 3D coordinate

# plot raw data
ax.scatter(x[:,1],x[:,-1],y)
x1=x[:,1]
x2=x[:,-1]
# plot fit plane
x1, x2 = np.meshgrid(np.linspace(-2,3.5,2), np.linspace(-3,2.5,2)) #important, generate Grid sampling points
h_x = theta[:,0] + theta[:,1] * x1 + theta[:,2] * x2 
ax.plot_wireframe(x1,x2,h_x,color='g',alpha = 0.4)	#plot wireframe,alpha is the transparency
ax.plot_surface(x1,x2,h_x,color='g',alpha = 0.5)	#plot plane
	
ax.set(xlabel='size',ylabel='bedrooms',zlabel='price') # 坐标轴

plt.show()
