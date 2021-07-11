import numpy as np

def encoder(y):
    '''
    one-hot-encoder could change the scatter data，let it be like 001,010,100, make the distance between different class more resonable
    '''
    result = []
    for i in y: # 1-10
        y_temp=np.zeros(10)
        y_temp[i-1] = 1 
        result.append(y_temp)
    return np.array(result)