import numpy as np  			
import pandas as pd				
import matplotlib.pyplot as plt

def dataPretreatment(path):
    data = pd.read_csv(path)
    data=np.array(data)           # get raw data by read_csv,and the type was Dataframe,so use np.array to transform it
    [row,col]=data.shape

    tempMean=np.mean(data,axis=0)
    tempStd=np.std(data,axis=0)
    flag=input("Dp you wanna a feature scaling? 1 to yes,else to skip\n")         # function as the description
    if flag=='1':
        data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
        print(data,'\n',tempMean,tempStd,data.shape)
    
    bias=np.ones(row)
    data=np.insert(data,0,bias,axis=1)              # insert the bias,which combines to the x0
    print(data)

    return data
    # return trainingSet,cvSet,testSet

