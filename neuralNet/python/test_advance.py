import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        target, cord = sample
        return torch.from_numpy(target), torch.from_numpy(cord)

class CityDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        lat = self.csv_file.iloc[idx, 0]
        lng = self.csv_file.iloc[idx, 1]
        country = self.csv_file.iloc[idx, 2]
        #target = self.csv_file.iloc[idx, 3]
        #target = np.array([target])
        #target = target.astype('int')
        lat = np.array([lat])
        lat = lat.astype('float32')
        lng = np.array([lng])
        lng = lng.astype('float32')
        cord = np.array([lat[0], lng[0]])
        cord = cord.astype('float32')
        if(country == "Russia"):
            target = np.array(1)
        else:
            target = np.array(0)
        sample = target, cord
        if self.transform:
            sample = self.transform(sample)

        return sample

class ner_net(nn.Module):
    def __init__(self):
        super(ner_net, self).__init__()
        D_in, H1, H2, H3, H4, H5, H6, H7, H8, D_out = 2, 10, 15, 15, 12, 10, 9, 8, 6, 2
        self.fc1 = torch.nn.Linear(D_in, H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.fc3 = torch.nn.Linear(H2, H3)
        self.fc4 = torch.nn.Linear(H3, H4)
        self.fc5 = torch.nn.Linear(H4, H5)
        self.fc6 = torch.nn.Linear(H5, H6)
        self.fc7 = torch.nn.Linear(H6, H7)
        self.fc8 = torch.nn.Linear(H7, H8)
        self.fc9 = torch.nn.Linear(H8, D_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x
def subset_ind(dataset, ratio: float):
    return np.random.choice(len(dataset), size=int(ratio*len(dataset)),replace=False)

#device=torch.device('cuda')
city_dataset = CityDataset(csv_file='worldcities.csv', transform=transforms.Compose([ToTensor()]))
#city_dataset = city_dataset.to(device)
val_inds = subset_ind(city_dataset, 0.3)

val_city_dataset = Subset(city_dataset, val_inds)
learn_city_dataset = Subset(city_dataset, [i for i in range(len(city_dataset)) if i not in val_inds])
batch_size = 60
learn_data = DataLoader(learn_city_dataset, batch_size, shuffle=True, num_workers=0)
val_data = DataLoader(val_city_dataset, batch_size, shuffle=False, num_workers=0)
data = DataLoader(city_dataset, batch_size = len(city_dataset), shuffle=False, num_workers=0)
for batch in data:
    Y, X = batch
    #X = batch["cord"]
    Y = Y.reshape(-1).type(torch.LongTensor)
datal = DataLoader(learn_city_dataset, batch_size = len(learn_city_dataset), shuffle=False, num_workers=0)
for batch in datal:
    Y_l, X_l = batch
    #X_l = batch["cord"]
    Y_l = Y_l.reshape(-1).type(torch.LongTensor)
datav = DataLoader(val_city_dataset, batch_size = len(val_city_dataset), shuffle=False, num_workers=0)
for batch in datav:
    Y_v, X_v = batch
    #X_v = batch["cord"]
    Y_v = Y_v.reshape(-1).type(torch.LongTensor)

#create neural net
net = ner_net()
#net.to(device, torch.float32)
loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
soft = torch.nn.Softmax()

learning_rate = 1e-2
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
print(batch_size)
loss = 1
losses = []
val_losses = []
val_accuracy = []
train_accuracy = []
i = 0
l_lambda = 0.02
l = 0
n_epoch = 18
for epoch in range(n_epoch):
    ep_losses = []
    ep_val_losses = []
    ep_val_accuracy = []
    ep_train_accuracy = []
    for batch in learn_data:
        y_batch, X_batch = batch
        #X_batch = batch["cord"]
        y_batch = y_batch.type(torch.LongTensor)
        #X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        #print(y_batch)
        
        y_pred = net(X_batch)
        _, predicted = torch.max(y_pred, 1)
        l = 0
        for i in range(len(y_batch)):
            if(predicted[i] == 0 and  y_batch[i] == 1):
                l = l + l_lambda
        loss = loss_fn(y_pred, y_batch.squeeze())/batch_size + l
        
        ep_losses.append(loss.item())
        
        ep_train_accuracy.append(torch.mean((y_batch.squeeze() == predicted).type(torch.float).clone().detach()).item())  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        for batch in val_data:
            y_batch, X_batch = batch
            #X_batch =batch["cord"]
            y_batch = y_batch.type(torch.LongTensor)
            #X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = net(X_batch)
            _, predicted = torch.max(y_pred, 1)
            l = 0
            for i in range(len(y_batch)):
                if(predicted[i] == 0 and  y_batch[i] == 1):
                    l = l + l_lambda
            loss = loss_fn(y_pred, y_batch.squeeze())/batch_size + l
            ep_val_losses.append(loss.item())
            _, predicted = torch.max(y_pred, 1)
            ep_val_accuracy.append(torch.mean((y_batch.squeeze() == predicted).type(torch.float).clone().detach()).item())
    plt.show()
    val_losses.append(np.mean(ep_val_losses))
    val_accuracy.append(np.mean(ep_val_accuracy))
    train_accuracy.append(np.mean(ep_train_accuracy))
    losses.append(np.mean(ep_losses))
    print("epoch = ",epoch)
    print("epoch val loss       = ", np.mean(ep_val_losses))
    print("epoch val accuracy   = ", np.mean(ep_val_accuracy))
    print("epoch train loss     = ", np.mean(ep_losses))
    print("epoch train accuracy = ", np.mean(ep_train_accuracy))
    print("learning_rate        = ", learning_rate)
    if((np.mean(ep_losses) < 0.2) and (learning_rate >= 5*1e-3 - 0.00001)):
        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    if((np.mean(ep_losses) < 0.05) and (learning_rate >= 1e-3 - 0.000001)):
        learning_rate = 1e-4
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    if((np.mean(ep_losses) < 0.03) and (learning_rate >= 1e-5)):
        learning_rate = 1e-5
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)


_, predicted_y_test = torch.max(net(X_v), 1)
_, predicted_y_learn = torch.max(net(X_l), 1)
#device=torch.device('cpu')
accuracy_learn_data = torch.mean((Y_l == predicted_y_learn).type(torch.float).clone().detach())
accuracy_val_data = torch.mean((Y_v == predicted_y_test).type(torch.float).clone().detach())
print("train accuracy = ", accuracy_learn_data.data.numpy())
print("valid accuracy = ", accuracy_val_data.data.numpy())


plt.plot(losses)
plt.plot(val_losses)
plt.legend(["traing losses", "validation losses"], loc ="lower right")
plt.xlabel('$epoch$')
plt.show()

plt.plot(val_accuracy)
plt.legend(["validation accuracy"], loc ="lower right")
plt.xlabel('$epoch$')
plt.show()

with torch.no_grad():
    h = 1

    X = X_v.numpy()
    Y = Y_v.numpy()
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    y_0 = net(X_v)
    _, y_pred = torch.max(y_0, 1)
    Y = y_pred.numpy()

    plt.scatter(X[:, 1], X[:, 0], c=Y, s=40, cmap=plt.cm.rainbow)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.show()