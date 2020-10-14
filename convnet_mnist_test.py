import numpy as np
import struct
from scipy import signal
from PIL import Image
from cv2 import cv2

np.set_printoptions(linewidth=np.inf)

def conv(x, W):
    numfilters, wrow, wcol = W.shape
    xrow, xcol = x.shape

    yrow = xrow - wrow + 1
    ycol = xcol - wcol + 1
    
    y = np.zeros([numfilters, yrow, ycol])
    
    for k in range(0, numfilters):
       
       filter = W[k, :, :]
       filter = np.rot90(np.squeeze(filter), 2)
       y[k, :, :] = signal.convolve2d(x, filter, mode="valid")
       
    return y

def ReLU(v):

    return np.maximum(v, 0)

def pool(x):
    
    numfilters, xrow, xcol = x.shape
    y = np.zeros([numfilters, xrow // 2, xcol // 2])

    for k in range(0, numfilters):

        filter = np.ones([2, 2]) / (2*2)
        image = signal.convolve2d(x[k, :, :], filter, mode="valid")
        y[k, :, :] = image[0::2, 0::2]
         
    return y

def softmax(x):
    
    ex = np.exp(x)
    y  = ex / np.sum(ex)

    return y

if __name__ == "__main__":
   
    ######################
    #     input          #
    ######################

    W1 = np.load("./weights/W1.npy")
    W5 = np.load("./weights/W5.npy")
    Wo = np.load("./weights/Wo.npy")  
    im_path = "/path/to/image"

    im = cv2.imread(im_path, 0)

    print(im, k)

    im = im / 255

    y1 = conv(im, W1)
    y2 = ReLU(y1)
    y3 = pool(y2)
    y4 = np.reshape(y3.transpose(1,2,0), (-1,1), order="F")
    v5 = np.matmul(W5, y4)
    y5 = ReLU(v5)
    v = np.matmul(Wo, y5)
    y = softmax(v)
    ind_max = np.argmax(y)
    print("Your number is", ind_max)
