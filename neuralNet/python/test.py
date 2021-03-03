import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

#get test and traning data
def get_part_data(data, ratio, A):
    leng = len(data)
    length = int(len(data)*ratio)
    int_valid = np.arange(1 + A, length + A, 1)
    int_learn = np.concatenate([np.arange(length + A, leng, 1),np.arange(1, A + 1, 1)])
    traning_data = data[int_valid]
    learn_data = data[int_learn]
    return traning_data, learn_data

#get data batch
def batch_gen(X, Y, batch_size):
    idx = np.random.randint(X.shape[0], size=batch_size)
    X_batch = X[idx]
    Y_batch = Y[idx]
    return Variable(torch.FloatTensor(X_batch)), Variable(torch.LongTensor(Y_batch))

def data_set_gen(X, Y, s):
    j = int(s/2.0)
    result = np.asarray(np.where(Y == 1))
    idx = np.random.randint(X.shape[0], size = j)
    #print(result)
    np.random.shuffle(result)
    #target_idx = np.random.randint(result, size=int(size/2.0))
    #print(int(size/2.0))
    target_idx = result[:j]
    #print(target_idx)
    #print(idx)
    i = np.concatenate((idx, target_idx[0]))
    #print(i)
    #for j in range(int(size/2.0)):
     #   idx[j] = target_idx[j]
    
    X_set = X[i]
    Y_set = Y[i]
    #print(Y_set.shape[0])
    return X_set, Y_set
#read data
sFile  = open("worldcities.csv", 'r')

str_file = sFile.readlines()
i = len(str_file)
lat = []
lng = []
country = []
target = []
for k in range(i):
    if(str_file[k] != '\n'):
        temp = str_file[k].replace(' ', '')
        temp = temp.replace(',', ' ')
        temp = temp.replace('\n', '')
        words = temp.split(' ')
        lat.append(words[0])
        lng.append(words[1])
        country.append(words[2])
        #target.append(words[-1])
        if(words[2] == "Russia"):
            target.append('1')
        else:
            target.append('0')

#data preparation
ratio = 0.2
lat_np = np.array(lat)
lng_np = np.array(lng)

leng = len(lng_np)
length = int(len(lng_np)*ratio)
A = random.randint(2, leng - length - 1)

ind = np.arange(1, leng, 1)
np.random.shuffle(ind)
ind_learn, ind_test = ind[length:], ind[:length]
#print(ind_learn)

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

#plt.figure(figsize=(10, 8))

#plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.rainbow)
#plt.show()

#create neural net
batch_size=100
D_in, H1, H2, H3,H4, H5, H6, H7, H8, D_out = 2, 10, 15, 15, 10, 9, 7, 5,10, 2
net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, H3),
    torch.nn.ReLU(),
    torch.nn.Linear(H3, H4),
    torch.nn.ReLU(),
    torch.nn.Linear(H4, H5),
    torch.nn.ReLU(),
    torch.nn.Linear(H5, H6),
    torch.nn.ReLU(),
    torch.nn.Linear(H6, H7),
    torch.nn.ReLU(),
    torch.nn.Linear(H7, D_out),
    torch.nn.ReLU()
)

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
#loss_fn = loss_function
Nn = 50
learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
losses = []
loss = 1
size = 1000
for j in range(Nn):
    X_t, Y_t = data_set_gen(X, Y, size)
    #print(Y_t)
    if(loss > -0.3):
        if(j > 10):
            learning_rate = 1e-4
            if(j > 30):
                learning_rate = 1e-5
                if(j > 380):
                    learning_rate = 1e-5
        optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
        for t in range(100):
            x_batch, y_batch = batch_gen(X_t, Y_t, batch_size)
            y_pred = net(x_batch)

            # loss
            loss = loss_fn(y_pred, y_batch)/batch_size
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#    else:
        #learning_rate = 1e-5
        #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate[j])

        #for t in range(200):
        #    x_batch, y_batch = batch_gen(X_t, Y_t, batch_size)
        #    y_pred = net(x_batch)
#
#            # loss
#            loss = loss_fn(y_pred, y_batch)/batch_size
#            losses.append(loss.item())

#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()

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
