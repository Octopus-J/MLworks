import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from gaussionPara import gaussionPara
from gaussion import gaussion
from getEpsilon import getEpsilon
from plotContour import plotContour

data=sio.loadmat('../data/ex8data1.mat')
x=data['X']               # 307*2,x[:,0] latency(ms), x[:,1] throughput(mb/s)
xval=data['Xval']         # 307*2
yval=data['yval']         # 307*1

mean,var=gaussionPara(x)

pval=gaussion(xval,mean,var)              # use the model parameters mean and var to caculate the probability of cross validation set
# pva2=gaussion(xval,mean,var,1)          # use this to check that two caculate way wether get the same results
# n=50
# print(pval[n],'\n',pva2[n])

epsilon,bestF1=getEpsilon(yval,pval)      # use cross validation set to get the best epsilon and F1

y_predict=gaussion(x,mean,var) 
for i in range(len(y_predict)):           # use the best epsilon and F1 to predict bad samples
    if y_predict[i]>epsilon:
        y_predict[i]=0
    else:
        y_predict[i]=1

plt.scatter(x[:,0],x[:,1],s=6)
plotContour(mean,var)
plt.scatter(x[np.where(y_predict==1),0],x[np.where(y_predict==1),1],s=80,marker='o',facecolors='none',edgecolors='r',linewidths=2)
plt.show()