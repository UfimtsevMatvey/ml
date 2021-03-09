import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import csv
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ner_net(nn.Module):
    def __init__(self):
        super(ner_net, self).__init__()
        D_in, H1, H2, H3,H4, H5, H6, H7, D_out = 2, 10, 15, 15, 10, 9, 7, 5, 2
        self.fc1 = torch.nn.Linear(D_in, H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.fc3 = torch.nn.Linear(H2, H3)
        self.fc4 = torch.nn.Linear(H3, H4)
        self.fc5 = torch.nn.Linear(H4, H5)
        self.fc6 = torch.nn.Linear(H5, H6)
        self.fc7 = torch.nn.Linear(H6, H7)
        self.fc8 = torch.nn.Linear(H7, D_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return F.log_softmax(x)

#get test and traning data
def get_part_data(data, ratio, A):
    leng = len(data)
    length = int(len(data)*ratio)
    int_valid = np.arange(1 + A, length + A, 1)
    int_learn = np.concatenate([np.arange(length + A, leng, 1),np.arange(1, A + 1, 1)])
    traning_data = data[int_valid]
    learn_data = data[int_learn]
    return traning_data, learn_data

def data_set_gen(X, Y, s):
    j = int(s/2.0)
    result = np.asarray(np.where(Y == 1))
    idx = np.random.randint(X.shape[0], size = j)
    np.random.shuffle(result)
    target_idx = result[:j]
    i = np.concatenate((idx, target_idx[0]))

    X_set = X[i]
    Y_set = Y[i]
    return X_set, Y_set

#get data batch
def batch_gen(X, Y, batch_size):
    idx = np.random.randint(X.shape[0], size=batch_size)
    X_batch = X[idx]
    Y_batch = Y[idx]
    return Variable(torch.FloatTensor(X_batch)), Variable(torch.LongTensor(Y_batch))

def data_preparation(lat, lng, target, ratio = 0.2):
    lat_np = np.array(lat)
    lng_np = np.array(lng)

    leng = len(lng_np)
    length = int(len(lng_np)*ratio)
    A = random.randint(2, leng - length - 1)

    ind = np.arange(1, leng, 1)
    np.random.shuffle(ind)
    ind_learn, ind_test = ind[length:], ind[:length]

    target_np = np.array(target)
    lat_test, lat_learn = lat_np[ind_test], lat_np[ind_learn]
    lng_test, lng_learn = lng_np[ind_test], lng_np[ind_learn]
    target_test, target_learn = target_np[ind_test], target_np[ind_learn]

    N = len(lat_learn)
    N_test = len(lat_test)

    D = 2
    K = 2
    X = np.zeros((N * K, D))
    X_test = np.zeros((N_test * K, D))
    Y = np.zeros(N * K, dtype='uint8')
    Y_test = np.zeros(N_test * K, dtype='uint8')

    for j in range(K):
        ix = range(int(N/2) * j,int(N/2) * (j + 1))
        ix_test = range(int(N_test/2) * j,int(N_test/2) * (j + 1))
        X[ix] = np.c_[lng_learn[ix], lat_learn[ix]]
        X_test[ix_test] = np.c_[lng_test[ix_test], lat_test[ix_test]]
        Y[ix] = target_learn[ix]
        Y_test[ix_test] = target_test[ix_test]

    return X, Y, X_test, Y_test

#read table
with open("worldcities.csv", 'r') as city_file:
    csv_file = csv.DictReader(city_file)
    lat = []
    lng = []
    target = []
    for row in csv_file:
        lat.append(row["lat"])
        lng.append(row["lng"])
        if(row["country"] == "Russia"):
            target.append('1')
        else:
            target.append('0')

#data preparation
X, Y, X_test, Y_test = data_preparation(lat, lng, target, ratio = 0.2)
#create neural net
net = ner_net()

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
losses = []
accuracy_test = []
accuracy_learn = []
batch_size=1500
for t in range(500):
    x_batch, y_batch = batch_gen(X, Y, batch_size)

    y_pred = net(x_batch)
    loss = loss_fn(y_pred, y_batch)/batch_size
    losses.append(loss.item())

    y_predicted_test = net(torch.tensor(X_test, dtype=torch.float32))
    y_predicted_learn = net(torch.tensor(X, dtype=torch.float32))

    _, predicted_y_test = torch.max(y_predicted_test, 1)
    _, predicted_y_learn = torch.max(y_predicted_learn, 1)

    Y_test = torch.tensor(Y_test, dtype=torch.float)
    Y_learn = torch.tensor(Y, dtype=torch.float)
    accuracy_learn_data = torch.mean((Y_learn == predicted_y_learn).type(torch.float).clone().detach())
    accuracy_learn.append(accuracy_learn_data.item())

    accuracy_test_data = torch.mean((Y_test == predicted_y_test).type(torch.float).clone().detach())
    accuracy_test.append(accuracy_test_data.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


y_predicted_test = net(torch.tensor(X_test, dtype=torch.float32))
y_predicted_learn = net(torch.tensor(X, dtype=torch.float32))
#print(type(y_predicted))
#print("y_predicted = ", y_predicted)
_, predicted_y_test = torch.max(y_predicted_test, 1)
_, predicted_y_learn = torch.max(y_predicted_learn, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

#print("y_val = ", Y)
#print("y_pred = ", predicted_y_test)

accuracy_test_data = torch.mean((Y_test == predicted_y_test).type(torch.float).clone().detach())
print("Test accuracy: %.5f" % accuracy_test_data)
#Accurancy on traning data
accuracy_traning_data = torch.mean((Y == predicted_y_learn).type(torch.float).clone().detach())
print("Traning accuracy: %.5f" % accuracy_traning_data)
#print losses
plt.plot(losses)
plt.plot(accuracy_test)
plt.plot(accuracy_learn)
#print(y_predicted)
#print(Y_val)

#print final result
h = 1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

Z = net(torch.autograd.Variable(grid_tensor))
Z = Z.data.numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.rainbow)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()