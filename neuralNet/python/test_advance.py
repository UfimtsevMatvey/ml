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
        target, cord = sample['target'], sample['cord']
        return {'target': torch.from_numpy(target),
                'cord': torch.from_numpy(cord)}

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
        target = self.csv_file.iloc[idx, 3]
        lat = np.array([lat])
        lat = lat.astype('float32')
        lng = np.array([lng])
        lng = lng.astype('float32')
        target = np.array([target])
        target = target.astype('float32')
        cord = np.array([lat[0], lng[0]])
        cord = cord.astype('float32')
        if(country == "Russia"):
            target = np.array([1])
        else:
            target = np.array([0])
        sample = {'target' : target, 'cord' : cord}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
        return x

def subset_ind(dataset, ratio: float):
    return np.random.choice(len(dataset), size=int(ratio*len(dataset)),replace=False)

device=torch.device('cuda')
city_dataset = CityDataset(csv_file='worldcities.csv', transform=transforms.Compose([ToTensor()]))
#city_dataset = city_dataset.to(device)
val_inds = subset_ind(city_dataset, 0.2)

val_city_dataset = Subset(city_dataset, val_inds)
learn_city_dataset = Subset(city_dataset, [i for i in range(len(city_dataset)) if i not in val_inds])
batch_size = 10
learn_data = DataLoader(learn_city_dataset, batch_size, shuffle=True, num_workers=0)
val_data = DataLoader(val_city_dataset, batch_size, shuffle=False, num_workers=0)



#create neural net
net = ner_net()
net.to(device, torch.float32)
loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 2*1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

loss = 1
losses = []
val_losses = []
val_accuracy = []
train_accuracy = []
i = 0
for epoch in range(50):
    ep_losses = []
    ep_val_losses = []
    ep_val_accuracy = []
    ep_train_accuracy = []
    for batch in learn_data:
        X_batch = batch["cord"]
        y_batch = batch["target"].reshape(-1).type(torch.LongTensor)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred = net(X_batch)
        loss = loss_fn(y_pred, y_batch)/batch_size
        ep_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for batch in val_data:
        X_batch =batch["cord"]
        y_batch =batch["target"].reshape(-1).type(torch.LongTensor)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred = net(X_batch)

        loss = loss_fn(y_pred, y_batch)/batch_size
        ep_val_losses.append(loss.item())
        _, predicted = torch.max(y_pred, 1)
        ep_val_accuracy.append(torch.mean((y_batch == predicted).type(torch.float).clone().detach()).item())  
    val_losses.append(np.mean(ep_val_losses))
    val_accuracy.append(np.mean(ep_val_accuracy))
    losses.append(np.mean(ep_losses))


plt.plot(losses)
plt.plot(val_losses)
plt.plot(val_accuracy)
plt.legend(["traing losses", "validation losses", "validation accuracy"], loc ="lower right")
plt.xlabel('$epoch$')
plt.show()

print(val_accuracy)