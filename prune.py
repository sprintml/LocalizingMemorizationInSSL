import numpy as np
import torch
import random
from model import ResNet9
import h5py


def generate_random_numbers(N, ratio):
    num_random_numbers = int(ratio * N)
    random_numbers = random.sample(range(N), num_random_numbers)

    return np.array(random_numbers)


neuron_index_to_train = 3
model = ResNet9(3, 10)
model.load_state_dict(torch.load('./model_weights.pth'))
ratio = 0.20
mode = 1
count = 0
pos = 'max'
with torch.no_grad():
    for name, param in model.named_parameters():
        if mode == 1:
            numberlist = generate_random_numbers(param.data.shape[0], ratio)
            for i in numberlist:
                param.data[i] = torch.zeros_like(param.data[i])
        elif mode == 0 and pos == 'max':
            data = h5py.File('./data/pos_per_layer.mat', 'r')
            numberlist = np.array(data['max'])
            for i in numberlist[count]:
                param.data[i] = torch.zeros_like(param.data[i])
        elif mode == 0 and pos == 'min':
            data = h5py.File('./data/pos_per_layer.mat', 'r')
            numberlist = np.array(data['min'])
            for i in numberlist[count]:
                param.data[i] = torch.zeros_like(param.data[i])
        elif mode == -1 and pos == 'max':
            data = h5py.File('./data/pos_all_layer.mat', 'r')
            numberlist = np.array(data['max'])
            for i in numberlist[count]:
                param.data[i] = torch.zeros_like(param.data[i])
        elif mode == -1 and pos == 'min':
            data = h5py.File('./data/pos_all_layer.mat', 'r')
            numberlist = np.array(data['min'])
            for i in numberlist[count]:
                param.data[i] = torch.zeros_like(param.data[i])
        count = count + 1

torch.save(model.state_dict(), './data/model_weights_ori_zeros_20.pth')
