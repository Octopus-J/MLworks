import numpy as np
from gaussionPara import gaussionPara

def gaussion(x,mean,var,multiFlag=0):
    m=len(x)
    n=x.shape[1]

    if multiFlag==0:                         # flag=0, means use original gaussion
        p=np.zeros(x.shape)
        for i in range(m):
            for j in range(n):
                temp1=1/( np.sqrt(2*np.pi) * np.sqrt(var[j]) )
                temp2=np.exp(-( ((x[i,j]-mean[j])**2) / (2*var[j]) ))
                p[i,j]=temp1*temp2

        tempP=p[:,0]
        for j in range(1,n):
            tempP=tempP*p[:,j]               # multiply all columns of p
        p=tempP                              # p m*1

    else:                                    # flag=1, means use multi-varian gaussion, note that in multi-varian gaussion the feature number must less than sample number(n<m)
        p=[]
        if n>m:
            print('The number of features\' MUST LESS THAN samples\'')
        else:
            #Sigma=((x-mean).T@(x-mean))/m   # n*n, if you can make sure that there is correlation between variables, use this
            Sigma=np.diag(var)               # n*n, use this normaly

            temp1=1/( ((2*np.pi)**(n/2)) * (np.sqrt( np.linalg.det(Sigma) )) )
            temp2=np.exp( (-1/2) * ((x-mean) @ (np.linalg.inv(Sigma)) @ (x-mean).T) )  # m*m
            temp2=np.diag(temp2)             # 1*m take out the diagonal element, important!
            p=temp1*temp2                    # p 1*m

    return p