import numpy as np
import scipy.io as sio
from sklearn import svm

np.set_printoptions(threshold=np.inf)
# Train data
data = sio.loadmat('../data/spamTrain.mat')
x=data['X']        # x 4000*1899, 1899 is number of features, corresponding to the 1899 words in vocab.txt
y=data['y']        # y 4000*1

# Test data
data = sio.loadmat('../data/spamTest.mat')
x_test=data['Xtest']   # x 1000*1899, 1899 is number of features, corresponding to the 1899 words in vocab.txt
y_test=data['ytest']   # y 1000*1, if y=1, means that's a spam

# choose a suitable C 
C_values=[0.01,0.03,0.1,0.3,1,3,10,30,100]

best_score=0
best_parameters=0

for c in C_values:
    svc=svm.SVC(C=c,kernel='linear')      # in this question, svm with linear kernel is enough
    svc.fit(x,y.flatten())
    score = svc.score(x_test,y_test.flatten())
    if score>best_score:
            best_score=score              # note the better score and parameters
            best_parameters=c
        
print('The best score is:',best_score,'The best C=',best_parameters)

svc_best=svm.SVC(C=best_parameters,kernel='linear')
svc_best.fit(x,y.flatten())               # trained the best model

score_train=svc_best.score(x,y.flatten())
score_test=svc_best.score(x_test,y_test.flatten())
print('The best model\'s scores in train set and test set is:',score_train,score_test)