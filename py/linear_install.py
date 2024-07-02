import random
import os
import struct

dataset_path = '/home/whisper/Programs/C/DeepLearning/datasets/linear'

W1 = 10
W2 = 20
b = 2
x1_list = [random.uniform(-100, 100) for _ in range(1000)]
x2_list = [random.uniform(-100, 100) for _ in range(1000)]
y_list = [W1*x1 + W2*x2 + b for x1, x2 in zip(x1_list, x2_list)]

# save to file
def saver():
    for i in range(1000):
        path = os.path.join(dataset_path, f'{i}.tsr')
        path_y = os.path.join(dataset_path, f'{i}_y.tsr')
        with open(path, 'wb') as f:
            bina = struct.pack('d', x1_list[i])
            f.write(bina)
            bina = struct.pack('d', x2_list[i])
            f.write(bina)
        with open(path_y, 'wb') as f:
            f.write(struct.pack('d', y_list[i]))

saver()