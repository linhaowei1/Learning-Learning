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

model = AutoEncoder()

def main():
    model.load_state_dict(torch.load('net_params.pkl'))

    # 要观看的数据
    view_data = train_data.data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
    encoded_data, _ = model(view_data)    # 提取压缩的特征值
    # x, y, z 的数据值
    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()
    values = train_data.targets[:200].numpy()  # 标签值
    for x,y,s in zip(X,Y,values):
        color = cm.rainbow(int(255*s/9))
        plt.text(x, y, str(s), fontsize = 12, color = color, style = "italic", weight = "light", verticalalignment='center', horizontalalignment='right', rotation=0)  # 标位子
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max())
    plt.show()
    
    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()   # continuously plot

    # original data (first row) for viewing
    view_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
    for i in range(N_TEST_IMG):
       a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

    _, decoded_data = model(view_data)
    for i in range(N_TEST_IMG):
        a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.draw()
    plt.pause(0)

if __name__=='__main__':
    main()
