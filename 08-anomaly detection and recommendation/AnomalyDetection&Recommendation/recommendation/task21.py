import numpy as np
import matplotlib.pyplot as plt
import groupParams as gp
import scipy.io as sio
import time
from scipy.optimize import minimize
from meanNormalize import meanNormalize
from costFunction import costFunction
from gradient import grd

# in this task, try to use collaborative filtering to predict movie ratings that were not rated

data1=sio.loadmat('../data/ex8_movies.mat')
r=data1['R']                                         # r 1682*943, binary data, means 1682 movies and 943 audiences，1 is rated, 0 not
y=data1['Y']                                         # y 1682*943, 0-5 corressponding the rate of movies given by audiences

data2=sio.loadmat('../data/ex8_movieParams.mat')
x=data2['X']                                         # x 1682*10, the features of each movie
theta=data2['Theta']                                 # theta 943*10, the weights data of users
# get movie names
movies=[]
with open('../data/movie_ids.txt','r',encoding='ANSI') as f:    # r read only 
    for line in f:
        movies.append(''.join(line.strip().split(' ')[1:]))     # split(' ') means split words by space, strip means delete space after splited
                                                                # join means group the words of movie names as a string

# use full data set or a small data (set the numbers artificialtly) set to train
n_features=int(data2['num_features'])                # 10              
n_movies=int(data2['num_movies'])                    # 1682     
n_users=int(data2['num_users'])                      # 943

# r_use=r[:n_movies,:n_users]                        # n_movies*n_users use front ** movies and front ** users to quickly train model
# y_use=y[:n_movies,:n_users]                        # n_movies*n_users
# x_use=x[:n_movies,:n_features]                     # n_movies*n_features use front ** features to modeling
# theta_use=theta[:n_users,:n_features]              # n_users*n_features
# movies_use=movies[:n_movies]

my_ratings=np.zeros((n_movies,1))                    # create your own rating list
my_ratings[9] = 5
my_ratings[12] = 4
my_ratings[56] = 3
my_ratings[78] = 5
my_ratings[278] = 4
my_ratings[568] = 5
my_ratings[980] = 3
my_ratings[1090] = 2

y=np.c_[y,my_ratings]                                # add your own data to the set
r=np.c_[r,my_ratings!=0]                             # r is a binary value ndarray

n_movies,n_users=y.shape

y_norm,y_mean=meanNormalize(y,r)

x_initial=np.random.random((n_movies,n_features))    # get random initial x and thetas
theta_initial=np.random.random((n_users,n_features))
param_initial=gp.groupParams(x_initial,theta_initial)
lamda=1

start=time.perf_counter()
res=minimize(fun=costFunction,                       # the minimize function will use TNC method to quickly convergence
            x0=param_initial,                        # in this optimize process, it will learned the best x and theta at same time
            args=(r,y_norm,n_movies,n_users,n_features,lamda),
            method='TNC',
            jac=grd,
            options={'maxiter':100})
end=time.perf_counter()
print('Optimization time=',(end-start)/60,'m',(end-start)%60,'s')

final_param=res.x                                    # get out final params, and divide into final x and final theta
final_x,final_theta=gp.ungroupParams(final_param,n_movies,n_users,n_features)

y_predict=(final_x@final_theta.T)                    # the last column is the prediction of your own movie tast
y_predict=y_predict[:,-1]
y_mean=y_mean.reshape(y_predict.shape)
y_predict=y_predict+y_mean                           # n_movies*1
index=np.argsort(y_predict)[::-1]                    # argsort will sort the y_predict from small to large and return the index, [::-1] will reverse it (large to samll)

print('Movie tast of your own:')
for i in range(len(my_ratings)):
    if my_ratings[i]!=0:
        print(my_ratings[i],' for movie ',movies[i])

print('These movies are recommended for you, enjoy!')
for j in range(0,10):
    print('Predict ',"{:2.2f}".format(y_predict[index[j]]),'  for',index[j],movies[index[j]])

# plt.imshow(y)                                      # plot raw data y
# plt.colorbar()
# plt.show()