import matplotlib.pyplot as plt
import numpy as np
import getData
from featureMapping import featureMapping
from costFunction import cost
from trainModel import trainModel

path='../ex5data1.mat'
x,y,cvx,cvy,testx,testy=getData.getData(path)   # x 12*1,y 12*1, cvx 21*1, cvy 21*1, testx 21*1, testy21*1

x=np.insert(x,0,1,axis=1)                       # 12*2
cvx=np.insert(cvx,0,1,axis=1)                   # 21*2
testx=np.insert(testx,0,1,axis=1)               # 21*2
lamda=1

final_theta=trainModel(x,y,lamda)

# plot fit line and raw data
fig,ax = plt.subplots()
ax.scatter(x[:,1],y)
ax.set(xlabel = 'change in water level(x)',ylabel = 'water flowing out of the dam(y)')

plt.plot(x[:,1],x@final_theta,c='r')
plt.show()

# plot learn curve
xrange=range(1,len(y)+1)
fig,ax=plt.subplots()

cost_m_train=[]
cost_m_cv=[]

for i in xrange:
    tempTheta=trainModel(x[:i,:],y[:i,:],lamda)
    tempTheta=tempTheta.reshape(2,1)
    train_i=cost(tempTheta,x[:i,:],y[:i,:],0)
    cv_i=cost(tempTheta,cvx,cvy,0)
    cost_m_train.append(train_i)                    # in the exam process, don't need regularization term
    cost_m_cv.append(cv_i)                          # use all cross validation set

plt.plot(xrange,cost_m_cv,label='cv cost')
plt.plot(xrange,cost_m_train,label='train cost')    # if you get a wrong curve, you can run this code serveral times to escape lacal minimum
plt.legend()
plt.xlabel('number of train examples')
plt.ylabel('error')

plt.show()

##########################################  part 2  #########################################################

# after the above analysis, it could be figure out that the linear model has a high bias, corresponding to underfitting
# so we need to use the featrue mapping technology to create more features
# path='../ex5data1.mat'
# x,y,cvx,cvy,testx,testy=getData.getData(path)   # x 12*1,y 12*1, cvx 21*1, cvy 21*1, testx 21*1, testy21*1
powers=6

mapping_x=featureMapping(x,powers)
mapping_cvx=featureMapping(cvx,powers)
mapping_testx=featureMapping(testx,powers)
lamda=1

final_theta=trainModel(mapping_x,y,lamda)

# plot fit curve and raw data
fig,ax = plt.subplots()
ax.scatter(x[:,1],y)
ax.set(xlabel = 'change in water level(x)',ylabel = 'water flowing out of the dam(y)')       # raw data

xx=np.linspace(-60,60,100)
x_=xx.reshape(100,1)
x_=np.insert(x_,0,1,axis=1)                      
x_=featureMapping(x_,powers)             # fit curve

plt.plot(xx,x_@final_theta,c='r')
plt.show()

# plot learn curve
xrange=range(1,len(y)+1)
fig,ax=plt.subplots()

cost_m_train=[]
cost_m_cv=[]
lamda=1                                 # here, you can change the value of lamda to see it's influence on the learn curve, to help you determine wether model is under/over fitting

for i in xrange:
    tempTheta=trainModel(mapping_x[:i,:],y[:i,:],lamda)
    tempTheta=tempTheta.reshape(mapping_x.shape[1],1)
    train_i=cost(tempTheta,mapping_x[:i,:],y[:i,:],0)
    cv_i=cost(tempTheta,mapping_cvx,cvy,0)
    cost_m_train.append(train_i)        # in the exam process, don't need regularization term
    cost_m_cv.append(cv_i)              # use all cross validation set

plt.plot(xrange,cost_m_cv,label='cv cost')
plt.plot(xrange,cost_m_train,label='train cost')    # if you get a wrong curve, you can run this code serveral times to escape lacal minimum
plt.legend()
plt.xlabel('number of train examples')
plt.ylabel('error')
plt.show()

# choose a suitable lamda
fig,ax=plt.subplots()
lamda_range=np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
cost_m_train=[]
cost_m_cv=[]

for i in lamda_range:
    tempTheta=trainModel(mapping_x,y,i)
    tempTheta=tempTheta.reshape(mapping_x.shape[1],1)
    train_i=cost(tempTheta,mapping_x,y,0)
    cv_i=cost(tempTheta,mapping_cvx,cvy,0)
    cost_m_train.append(train_i)        # in the exam process, don't need regularization term
    cost_m_cv.append(cv_i)              # use all cross validation set

plt.plot(lamda_range,cost_m_cv,label='cv cost')
plt.plot(lamda_range,cost_m_train,label='train cost')             # if you get a wrong curve, you can run this code serveral times to escape lacal minimum
plt.legend()
plt.xlabel('value of lamda')
plt.ylabel('error')

plt.show()

print('A probably lamda is:',lamda_range[np.argmin(cost_m_cv)])   # use the index of cv set's minimum value to get corresponding lamda value

lamda=lamda_range[np.argmin(cost_m_cv)]