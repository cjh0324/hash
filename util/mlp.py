#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


# In[2]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 8192)
        self.fc5 = nn.Linear(8192, 18)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x


# In[3]:


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:,0].values
        self.y = df.iloc[:,1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = self.x[index]
        x = list(x)[:-6]
        x = list(map(int, x))
        # x = [1 if xi == 1 else -1 for xi in x]
        x = torch.FloatTensor(x)
        y = torch.LongTensor(self.y)[index]
        return x, y

    def __len__(self):
        return self.length


# In[4]:


print(os.getcwd())


# In[5]:


full_dataset = CustomDataset("txt/data_Xeon.csv")
# full_dataset = CustomDataset("C:/Users/1998b/Google 드라이브/학부/4학년 2학기/hash/txt/data_Xeon.csv")
# full_dataset = CustomDataset("/content/drive/MyDrive/학부/4학년 2학기/hash/txt/data.csv")
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
# test_dataloader = DataLoader(test_dataset, batch_size=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
print(model)


# In[6]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[ ]:


losses = []

indices = torch.randperm(len(train_dataset))[:2097152]
sampler = SubsetRandomSampler(indices)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True, sampler=sampler)

for epoch in range(200):
    cost = 0.0

    # indices = torch.randperm(len(train_dataset))[:131072]
    # sampler = SubsetRandomSampler(indices)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True, sampler=sampler)

    # for i, batch in enumerate(tqdm(train_dataloader)):
    for x, y in train_dataloader:
        # x = batch[0].to(device)
        # y = batch[1].to(device)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        output = model(x)

        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        cost += loss

        # print(f"Epoch : {epoch+1:4d}, Loss : {loss:.3f}")

    cost = cost / len(train_dataloader)
    losses.append(cost.item())

    # if (epoch + 1) % 10 == 0:
    print(f"Epoch : {epoch+1:4d}, Cost : {cost:.3f}")


# In[16]:


plt.plot(losses, 'r')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Xeon Gold 6136')
plt.savefig('epoch 1000.png')
plt.show()

indices = torch.randperm(len(test_dataset))[:2000000]
sampler = SubsetRandomSampler(indices)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, sampler=sampler)

# GET OUTPUT
with torch.no_grad():
    model.eval()
    
    total = 0
    correct = 0

    for i, batch in enumerate(tqdm(test_dataloader)):
        x = batch[0].to(device)
        y = batch[1].to(device)
        # x = x.to(device)
        # y = y.to(device)
        output = model(x)
        y_pred = torch.argmax(output, dim=1).to(device)
        total += 1
        # print(x, y, y_pred)
        if (y == y_pred): 
            correct += 1

print('--------------')
print('Accuracy: ', correct / total * 100)

