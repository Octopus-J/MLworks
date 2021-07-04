import numpy as np

def featureMapping(x,powers):    # feature mapping was designed to solve the linear inseparable problem in logistic regression by intoduced the high-order term.
    data=x[:,1:x.shape[1]+1]     # take out all features
    feature1=data[:,0]
    feature2=data[:,1]

    k=0
    for i in range(0,powers+1):    # gei the column size of data after mapping
        for j in range(0,i+1):
            k+=1

    data=np.zeros((len(feature1),k))
    k=0
    for i in range(0,powers+1):    # introduce the high-order iterm, i is the order of feature1,j feature2 
        for j in range(0,i+1):
            data[:,k]=np.power(feature1,i-j)*np.power(feature2,j)
            k+=1
    return data
