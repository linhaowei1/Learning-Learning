import torch
from torchvision import datasets, transforms
import torch.nn.functional as F 
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
# define hyper parameters
EPOCHS = 30
BATCH_SIZE = 128
N_TEST_IMG = 10
# load data
train_data = datasets.MNIST(
    root='./data', train=True, download=True, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# model
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.view(-1,28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train():
    lossfunc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(EPOCHS):
        train_loss = 0.0
        for step, (x, target) in enumerate(train_loader):
            data = x.view(-1, 28*28)
            encoded, decoded = model(data)
            loss = lossfunc(decoded, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()*data.size(0)
            
        #train_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.8f}'.format(epoch+1, train_loss))
    torch.save(model.state_dict(), 'net_params.pkl')

model = AutoEncoder()

def main():
    train()

if __name__=='__main__':
    main()