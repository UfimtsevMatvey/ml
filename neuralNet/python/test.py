
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

batch_size=32
N, D_in, H, D_out = 64, 2, 25, 2

# Use the nn package to define our model and loss function.
two_layer_net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)

N = 1000
D = 2
K = 2
X = np.zeros((N * K, D))
X_val = np.zeros((N * K, D))
Y = np.zeros(N * K, dtype='uint8')
Y_val = np.zeros(N * K, dtype='uint8')
for j in range(K):
    ix = range(N * j,N * (j + 1))
    X[ix] = np.c_[np.random.randn(N)*5 + j*10, np.random.randn(N)*5 + j*10]
    X_val[ix] = np.c_[np.random.randn(N)*5.5 + j*10, np.random.randn(N)*5.4 + j*10]
    Y[ix] = j
    Y_val[ix] = j

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
for t in range(10000):
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


#print final result
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