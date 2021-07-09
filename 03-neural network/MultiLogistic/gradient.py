import numpy as np
import sigmoidFunction

def grd(gtheta,x,y,lamda):               # gtheta 401*1,x 5000*401,y 5000*1,notice that theta must be the first parameter because of the claim of minimize.
    m=len(y)
    gtheta=gtheta.reshape(401,1)
    _theta=gtheta[1:]
    _theta=np.insert(_theta,0,0)         # don't punish the theta0, and don't change the origin data,_theta 1*401

    sig=sigmoidFunction.sig(x@gtheta)    # sig 5000*1

    para1=(x.T)@(sig-y)/m                # para1 401*1
    para2=lamda*_theta/m                 # para2 1*401
    para2=para2.reshape((401,1))         # para2 401*1

    return para1+para2                   # 401*1