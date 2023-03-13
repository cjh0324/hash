#-*- coding:utf-8 -*-
f2 = open("txt/wrong_idx", "w")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 18)
        self.fc5 = nn.Linear(4096, 18)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        return x

model = torch.load('util/model')
model.eval()

perms = [[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]]

base_seq = ['1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

wrong = 0
correct = 0

base_arr = []
with open("txt/cmp1.csv") as cmp1:
    for line in cmp1:
        base_arr.append(int(line[13:]))

with open("txt/data_Xeon.csv") as bookfile:
    for i, line in enumerate(tqdm(bookfile)):
    # for line in bookfile:
        seq_idx = line[:16]
        idx = line[16:28]
        slice = int(line[35:])
        if (int(idx) == 0):
            perm_mat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for k in range(1, 16):
                if (seq_idx[k] != base_seq[k]):
                    for j in range(0, 12):
                        perm_mat[j] = perm_mat[j] ^ perms[k][j]
            perm = int("".join(str(x) for x in perm_mat), 2)
        idx = int(idx, 2) ^ perm

        # Deep learning Ver.
        # idx = bin(idx)[2:].zfill(12)
        # input = list(idx)
        # input = list(map(int, input))
        # input = [1 if xi == 1 else -1 for xi in input]
        # input = torch.FloatTensor(input)
        # predict = torch.argmax(model(input)).item()    

        # Arr Ver.
        predict = base_arr[idx]

        if (predict != slice):
            wrong += 1
            f2.write(str(i)+' ' + str(seq_idx) + ' ' + str(slice) + ' ' + str(predict) + '\n')
        else:
            correct += 1

        # if (correct == 100000):
        #     break

print(wrong, correct)
f2.close()