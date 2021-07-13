import numpy as np

def featureMapping(x,powers):      # feature mapping was designed to solve the linear inseparable problem in logistic regression by intoduced the high-order term.
    '''return a normalized mapping feature matrix'''
    feature=x[:,1]
    data=np.zeros((x.shape[0],powers+1))                # take out all features

    for i in range(0,powers+1):    # introduce the high-order iterm, i is the order of feature1,j feature2 
        data[:,i]=np.power(feature,i)

    # normalize
    means=np.mean(data,axis=0)     # average value of each feature
    std=np.std(data,axis=0)        # standard deviation of each feature

    data=(data-means)/std           
    data[:,0]=1                    # the first column is all one,don't normalize
    return data