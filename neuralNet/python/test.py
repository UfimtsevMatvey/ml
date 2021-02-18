
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def batch_gen(X, y, batch_size=128):
    idx = np.random.randint(X.shape[0], size=batch_size)
    X_batch = X[idx]
    y_batch = y[idx]
  
    return Variable(torch.FloatTensor(X_batch)), Variable(torch.LongTensor(y_batch))

N, D_in, H, D_out = 64, 2, 50, 2

# Use the nn package to define our model and loss function.
two_layer_net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)

N = 1000
D = 2
K = 2
X = np.zeros((N * K, D))
Y = np.zeros(N * K, dtype='uint8')
for j in range(K):
    ix = range(N * j,N * (j + 1))
    X[ix] = np.c_[np.random.randn(N)*5.5 + j*10, np.random.randn(N)*5.4 + j*10]
    Y[ix] = j

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


for t in range(10000):
    x_batch, y_batch = batch_gen(X, Y)
    
    # forward
    y_pred = two_layer_net(x_batch)

    # loss
    loss = loss_fn(y_pred, y_batch)
    #print('{} {}'.format(t, loss.data))

    # ЗАНУЛЯЕМ!
    optimizer.zero_grad()

    # backward
    loss.backward()

    # ОБНОВЛЯЕМ! 
    optimizer.step()


h = 0.02
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
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.rainbow)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()

y_predicted = two_layer_net(torch.tensor([0,0], dtype=torch.float32))
print(y_predicted)