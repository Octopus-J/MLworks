import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from kmeans import kmeans
from skimage import io


# In this task, use kmeans to compress image
# it should be noted that the bird figure is a 24-bit true color image
# means every pix has red(R), green(G), blue(B) three properties, each range is 0-255
# the bird image has been transformed into a mat file, bird_small.mat, just load it, it's 128*128*3, 128*128 pixels
# use kmeans to find 16 colors to compress figure

data=sio.loadmat('../data/bird_small.mat')              # 128*128*3
image=io.imread('../data/bird_small.png')
figureA=data['A']

figureA=figureA/255                                     # in plt.imshow(), it use [0-1] to present the [0-255]
figureA=figureA.reshape(-1,3)                           # reshape figureA to 2D, 16384*3,16384=128*128

k=16
iters=20

centers,class_all,center_note=kmeans(figureA,k,iters)   # find k classes and centers

compressedA=np.zeros(figureA.shape)

for i in range(k):
    print(i)
    compressedA[class_all==i]=centers[i,:]              # compress A

compressedA=compressedA.reshape(128,128,3)

fig,ax1=plt.subplots()
ax1=plt.imshow(data['A'])                               # plot initial figure

fig,ax2=plt.subplots()
ax2=plt.imshow(compressedA)                             # plot compressed figure

plt.show()