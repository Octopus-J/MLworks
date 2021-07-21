import numpy as np

def getEpsilon(yval,pval):
    bestF1=0
    bestEpsilon=0

    for epsilon in np.linspace(min(pval),max(pval),1000):
        y_predict=np.zeros(yval.shape)
        y_predict[pval<=epsilon]=1                                  # p less than epsilon should be set to 1, the abnormals

        precisions=np.sum(y_predict[yval==1])/np.sum(y_predict)     # precision rate, true positive / (true positive + fake positive)
        recall=np.sum(y_predict[yval==1])/np.sum(yval)              # recall rate, true positive / (true positive + fake negative)
        tempF1=(2*precisions*recall)/(precisions+recall)

        if tempF1>bestF1:                                           # use F1 score to judge an epsilon weather is suitable
            bestF1=tempF1*1
            bestEpsilon=epsilon*1


    return bestEpsilon,bestF1