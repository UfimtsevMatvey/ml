import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_part_data(data, ratio):
    leng = len(data)
    length = int(len(data)*ratio)
    ind_valid = np.arange(1, length, 1)
    int_learn = np.arange(length, leng, 1)
    traning_data = data[ind_valid]
    learn_data = data[int_learn]
    return traning_data, learn_data


def batch_gen(X, y, batch_size=128):
    idx = np.random.randint(X.shape[0], size=batch_size)
    X_batch = X[idx]
    y_batch = y[idx]
  
    return Variable(torch.FloatTensor(X_batch)), Variable(torch.LongTensor(y_batch))

#read data
sFile  = open("worldcities.csv", 'r')

instrs = sFile.readlines()
i = len(instrs)
print(instrs[1])
lat = []
lng = []
country = []
target = []
for k in range(i):
    #begin
    if(instrs[k] != '\n'):
        #begin
        instrTemp = instrs[k].replace(' ', '')
        instrTemp = instrTemp.replace(',', ' ')
        instrTemp = instrTemp.replace('\n', '')
        #instrTemp = re.sub('\s+',' ',instrTemp)
        words = instrTemp.split(' ')
        lat.append(words[0])# = words[0]
        lng.append(words[1])
        country.append(words[2])
        target.append(words[-1])
        #end
    #end
ratio = 0.15
lat_np = np.array(lat)
lng_np = np.array(lng)
target_np = np.array(target)
lat_test, lat_learn = get_part_data(lat_np, ratio)
lng_test, lng_learn = get_part_data(lng_np, ratio)
target_test, target_learn = get_part_data(target_np, ratio)
print(target_test)
N = len(lat_learn)
print(lat_learn[0])

#create neural net
batch_size=10
N, D_in, H, D_out = 64, 2, 25, 2

# Use the nn package to define our model and loss function.
two_layer_net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)
#data preparation
N = len(lat_learn)
N_val = len(lat_test)
D = 2
K = 2
X = np.zeros((N * K, D))
X_val = np.zeros((N_val * K, D))
Y = np.zeros(N * K, dtype='uint8')
Y_val = np.zeros(N_val * K, dtype='uint8')

for j in range(K):
    ix = range(int(N/2) * j,int(N/2) * (j + 1))
    ix_val = range(int(N_val/2) * j,int(N_val/2) * (j + 1))
    X[ix] = np.c_[lng_learn[ix], lat_learn[ix]]
    X_val[ix_val] = np.c_[lng_test[ix_val], lat_test[ix_val]]
    Y[ix] = target_learn[ix]
    Y_val[ix_val] = target_test[ix_val]
"""
for j in range(K):
    ix = range(N * j,N * (j + 1))
    X[ix] = np.c_[np.random.randn(N)*5 + j*10, np.random.randn(N)*5 + j*10]
    X_val[ix] = np.c_[np.random.randn(N)*5.5 + j*10, np.random.randn(N)*5.4 + j*10]
    Y[ix] = j
    Y_val[ix] = j
"""

plt.figure(figsize=(10, 8))

plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.rainbow)
plt.show()
# create dummy data with 3 samples and 784 features
x_batch = torch.tensor(X[:2], dtype=torch.float32)
y_batch = torch.tensor(Y[:2], dtype=torch.float32)

# compute outputs given inputs, both are variables
y_predicted = two_layer_net(torch.tensor(x_batch, dtype=torch.float32))

print(y_predicted)
print(x_batch)

batch_gen(X, Y)[1].shape

print(two_layer_net.forward(batch_gen(X,Y)[0]))


loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(two_layer_net.parameters(), lr=learning_rate)
losses = []
for t in range(4000):
    x_batch, y_batch = batch_gen(X, Y, batch_size)

    y_pred = two_layer_net(x_batch)

    # loss
    loss = loss_fn(y_pred, y_batch)/batch_size
    #print('{} {}'.format(t, loss.data))
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_predicted = two_layer_net(torch.tensor(X_val, dtype=torch.float32))
y_pred = two_layer_net(torch.tensor(X, dtype=torch.float32))
print(type(y_predicted))
print("y_predicted = ", y_predicted)
_, predicted_y_test = torch.max(y_predicted, 1)
_, predicted_y_train = torch.max(y_pred, 1)
Y_val = torch.tensor(Y_val, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)
#predicted_y_test = np.array(y_predicted > 0.5)

#isinstance(predicted_y_test, np.ndarray)
#predicted_y_test.shape == Y_val.shape
#np.in1d(predicted_y_test, Y_val).all()

#accuracy = torch.mean(torch.tensor(labels == predicted, dtype=torch.float))
#print(type(predicted_y_test))
print("y_val = ", Y)
print("y_pred = ", predicted_y_test)
#accuracy = np.mean(Y_val == predicted_y_test)
#Accurancy on test data
accuracy_test_data = torch.mean((Y_val == predicted_y_test).type(torch.float).clone().detach())
print("Test accuracy: %.5f" % accuracy_test_data)
#Accurancy on traning data
accuracy_traning_data = torch.mean((Y == predicted_y_train).type(torch.float).clone().detach())
print("Traning accuracy: %.5f" % accuracy_traning_data)
#print losses
plt.plot(losses)

print(y_predicted)
print(Y_val)
#print final result
h = 1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

Z = two_layer_net(torch.autograd.Variable(grid_tensor))
Z = Z.data.numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)
plt.scatter(X_val[:, 0], X_val[:, 1], c=predicted_y_test, s=40, cmap=plt.cm.rainbow)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
