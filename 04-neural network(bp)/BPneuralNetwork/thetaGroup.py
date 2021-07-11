import numpy as np

# In this question, we need to use the minimize function,it requires only one theta parameter in cost and gradient function
# So the two functions below could group/ungroup theta1 and theta2

def thetaGroup(theta1,theta2):
    return np.append(theta1.flatten(),theta2.flatten())

def thetaUnGroup(theta):
    theta1=theta[:25*401].reshape(25,401)
    theta2=theta[25*401:].reshape(10,26)
    return theta1,theta2