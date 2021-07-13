import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def getData(path):
    if path==' ':
        print("You didn't select the data path,try again.")
    else:
        data=sio.loadmat(path)
        trainX=data['X']                   # 12*1
        trainy=data['y']                   # 12*1
        cvX=data['Xval']
        cvy=data['yval']
        testX=data['Xtest']
        testy=data['ytest']

        return trainX, trainy, cvX, cvy, testX, testy