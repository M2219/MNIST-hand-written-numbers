import numpy as np
import matplotlib.pyplot as plt
import struct


with open('train-labels-idx1-ubyte', 'rb') as f:
# with open('t10k-labels-idx1-ubyte', 'rb') as f:
    data = f.read(8)
    magic_number, number_of_item = struct.unpack('>II', data)
    print(magic_number, number_of_item)

    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((number_of_item, 1))

print(data)
print(data.shape)