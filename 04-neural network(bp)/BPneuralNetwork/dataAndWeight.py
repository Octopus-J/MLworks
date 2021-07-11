import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def getData(path):
    if path==' ':
        print("You didn't select the data path,try again.")
    else:
        data=sio.loadmat(path)
        rawX=data['X']                   # 5000*400
        rawX=np.insert(rawX,0,1,axis=1)  # 5000*401
        rawy=data['y']                   # 5000*1

        flag=0                                                          # flag to control wheather show the data,0 to skip
        if flag==1:
            sampleNumber = np.random.choice(len(rawX),100)              # selected ramdomly 100 samlpes from original data
            images = rawX[sampleNumber,:]                               # get out the 100 samples   	
            print(images.shape)

            fig,ax = plt.subplots(ncols=10,nrows=10,figsize=(5,5),sharex=True,sharey=True)    # 100 pictures in total(10*10), each one is 5×5, share x,y axis
            plt.xticks([])                                              # hide x,y labels
            plt.yticks([])
            
            for r in range(10):               # row
                for c in range(10):           # column
                    ax[r,c].imshow(images[10 * r + c].reshape(20,20).T,cmap='gray_r')
                
            plt.show()

    return rawX,rawy
