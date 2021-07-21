import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from gaussionPara import gaussionPara
from gaussion import gaussion
from getEpsilon import getEpsilon
from plotContour import plotContour

data=sio.loadmat('../data/ex8data2.mat')
x=data['X']               # 1000*11,x[:,0] latency(ms), x[:,1] throughput(mb/s)
xval=data['Xval']         # 100*11
yval=data['yval']         # 100*1

mean,var=gaussionPara(x)

pval=gaussion(xval,mean,var,1)            # use the model parameters mean and var to caculate the probability of cross validation set

epsilon,bestF1=getEpsilon(yval,pval)      # use cross validation set to get the best epsilon and F1

y_predict=gaussion(x,mean,var,1) 
for i in range(len(y_predict)):           # use the best epsilon and F1 to predict bad samples
    if y_predict[i]>epsilon:
        y_predict[i]=0
    else:
        y_predict[i]=1

print('epsilon=',epsilon,'F1=',bestF1,'\n','predictions=\n',y_predict,np.sum(y_predict))