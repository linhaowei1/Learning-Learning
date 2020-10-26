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
N_TEST_IMG = 10
RANGE = 5
# model
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

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
        decoded = self.decoder(x)
        return decoded

model = Decoder()

def main():
    model.load_state_dict(torch.load('./autoencoder/net_params.pkl'))
    x = (torch.rand(N_TEST_IMG**2,2)-0.5)*(RANGE*2)
    # initialize figure
    f, a = plt.subplots(N_TEST_IMG, N_TEST_IMG, figsize=(5, 2))
    plt.ion()   # continuously plot

    decoded_data = model(x)
    for i in range(N_TEST_IMG):
        for j in range(N_TEST_IMG):
            a[i][j].imshow(np.reshape(decoded_data.data.numpy()[j+i*N_TEST_IMG], (28, 28)), cmap='gray')
            a[i][j].set_xticks(()); a[i][j].set_yticks(())
            plt.draw()
    plt.pause(0)

if __name__=='__main__':
    main()
