	import numpy as np
	import matplotlib.pyplot as plt
	import struct


	with open('train-images-idx3-ubyte', 'rb') as f:
	# with open('t10k-images-idx3-ubyte', 'rb') as f:
	    data = f.read(16)
	    magic_number, number_of_image, rows, cols = struct.unpack('>IIII', data)
	    print(magic_number, number_of_image, rows, cols)

	    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
	    data = data.reshape((number_of_image, rows, cols))
	    # data = data.reshape((number_of_image,  rows*cols, 1))

	plt.imshow(data[0,:,:], cmap='gray')
	plt.show()
