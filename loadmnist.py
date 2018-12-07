"""
MNIST数据的读取
"""
import os
import struct
import numpy as np
import torch as tc

def load_mnist(path, kind='train', ret='np', rtype='bin'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    if(rtype == "bin"):
        images = np.minimum(images, 1)
    if ret == 'tc':
        return tc.from_numpy(images), tc.from_numpy(labels)
    return images, labels
