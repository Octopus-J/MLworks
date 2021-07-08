import numpy as np
import sigmoidFunction

def grd(theta,x,y,lamda):                # theta 401*1,x 5000*401,y 5000*1,notice that theta must be the first parameter because of the claim of minimize.
    m=len(y)
    _theta=theta[1:]
    _theta=np.insert(_theta,0,0,axis=0)  # don't punish the theta0, and don't change the origin data,_theta 401*1

    sig=sigmoidFunction.sig(x@theta)     # sig 1*5000
    sig=sig.reshape((5000,1))            # sig 5000*1

    para1=(x.T)@(sig-y)/m                # para1 401*1
    para2=lamda*_theta/m                 # para2 1*401
    para2=para2.reshape((401,1))         # para2 401*1

    return para1+para2                   # 401*1