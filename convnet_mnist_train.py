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


def sub2ind(array_shape, rows, cols):

    return rows*array_shape[1] + cols





###################################
#     minibach algorithm          #
###################################

def mnist_conv(W1, W5, Wo, input_images, input_labels):

    alpha = 0.01   # learning rate
    beta = 0.95    # momentum constant

    moment1 = np.zeros(W1.shape)
    moment5 = np.zeros(W5.shape)
    momento = np.zeros(Wo.shape)
    
    N = len(input_labels)
    bsize = 100
    blist = list(range(0, (N - bsize + 1), bsize))

    ###################################
    #     one epoch loop              #
    ###################################

    for batch in range(0, len(blist)):

        dw1 = np.zeros(W1.shape)
        dw5 = np.zeros(W5.shape)
        dwo = np.zeros(Wo.shape)

        ###################################
        #     mini batch loop             #
        ###################################

        begin = blist[batch]

        for k in range(begin, begin + bsize):

            ## forward
            x = input_images[k, :, :]
            y1 = conv(x, W1)
            y2 = ReLU(y1)
            y3 = pool(y2)
            y4 = np.reshape(y3.transpose(1,2,0), (-1,1), order="F")
            v5 = np.matmul(W5, y4)
            y5 = ReLU(v5)
            v = np.matmul(Wo, y5)
            print(v)
            y = softmax(v)
            print(y)

            ## one-hot encoding

            d = np.zeros([10, 1])
            d[sub2ind(d.shape, input_labels[k], 0)] = 1

            ## back propagation

            e = d - y
            delta = e # cross enthropy

            e5 = np.matmul(Wo.T, delta)

            y5_temp = y5
            y5_temp[y5_temp > 0] = 1
            y5_temp[y5_temp <= 0] = 0

            delta5 = np.multiply(y5_temp, e5)

            e4 = np.matmul(W5.T, delta5)
            e3 = np.reshape(e4, y3.shape, order="C").transpose(0,2,1)
            e2 = np.zeros(y2.shape)
            W3 = np.ones(y2.shape) / (2 * 2)

            for c in range(0, 20):
                e2[c, :, :] = np.multiply(np.kron(e3[c, : ,:], np.ones((2, 2))), W3[c, : ,:])

            y2_temp = y2
            y2_temp[y2_temp > 0] = 1
            y2_temp[y2_temp <= 0] = 0

            delta2 = np.multiply(y2_temp, e2)

            delta1_x = np.zeros(W1.shape)

            for c in range(0, 20):
                delta1_x[c, :, :] = signal.convolve2d(x[:, :], np.rot90(delta2[c, :, :], 2), mode="valid")


            ## learning rule

            dw1 = dw1 + delta1_x
            dw5 = dw5 + np.matmul(delta5, y4.T)
            dwo = dwo + np.matmul(delta, y5.T) 

            

        ## update weights
        
        dw1 = dw1 / bsize
        dw5 = dw5 / bsize
        dwo = dwo / bsize    

        moment1 = alpha * dw1 + beta * moment1        
        W1 = W1 + moment1

        moment5 = alpha * dw5 + beta * moment5        
        W5 = W5 + moment5

        momento = alpha * dwo + beta * momento        
        Wo = Wo + momento

        
    return W1, W5, Wo 



if __name__ == "__main__":
   
    ######################
    #     input          #
    ######################

    with open('train-images-idx3-ubyte', 'rb') as f:

        data = f.read(16)
        magic_number, number_of_image, rows, cols = struct.unpack('>IIII', data)

        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        input_images = data.reshape((number_of_image, rows, cols))
        


    with open('train-labels-idx1-ubyte', 'rb') as f:

        data = f.read(8)
        magic_number, number_of_item = struct.unpack('>II', data)

        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        input_labels = data.reshape((number_of_item, 1))

    input_images = input_images / 255

    W1 = 1e-2 * np.random.randn(20 , 9, 9)   
    W5 = (2 * np.random.rand(100, 2000) - 1) * np.sqrt(6) / np.sqrt(100 + 2000) * 0
    Wo = (2 * np.random.rand(10, 100) - 1) * np.sqrt(6) / np.sqrt(10 + 100) * 0

    for epoch in range(0, 3):
        break
        W1, W5, Wo = mnist_conv(W1, W5, Wo, input_images[0:50000, :, :], input_labels[0:50000])
        

    np.save("W1.npy", W1)
    np.save("W5.npy", W5)
    np.save("Wo.npy", Wo)
   
    ## Test 


    with open('t10k-images-idx3-ubyte', 'rb') as f:

        data = f.read(16)
        magic_number, number_of_image, rows, cols = struct.unpack('>IIII', data)

        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        input_images_t10k = data.reshape((number_of_image, rows, cols))
        


    with open('t10k-labels-idx1-ubyte', 'rb') as f:

        data = f.read(8)
        magic_number, number_of_item = struct.unpack('>II', data)

        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        input_labels_t10k = data.reshape((number_of_item, 1))


    W1 = np.load("./weights/W1.npy")
    W5 = np.load("./weights/W5.npy")
    Wo = np.load("./weights/Wo.npy")  

   
    input_images_test = input_images_t10k / 255
    input_labels_test = input_labels_t10k

    acc = 0

    N = len(input_labels_test)

    for k in range(0, N):
        break
        x = input_images_test[k, :, :]
        y1 = conv(x, W1)
        y2 = ReLU(y1)
        y3 = pool(y2)
        y4 = np.reshape(y3.transpose(1,2,0), (-1,1), order="F")
        v5 = np.matmul(W5, y4)
        y5 = ReLU(v5)
        v = np.matmul(Wo, y5)
        y = softmax(v)

        ind_max = np.argmax(y)

        if ind_max == input_labels_test[k]:
            acc = acc + 1

    acc = acc / N
    print("Accuracey is = ", acc)

    ###################################
