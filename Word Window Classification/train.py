from model import *
from torch.utils.data import DataLoader
from functools import partial

loader = DataLoader(datasets, batch_size = batch_size, shuffle = shuffle,
    collate_fn = partial(custom_collate_fn, window_size = window_size, word2inx = word2inx)
)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

train(epoch_num = 5000, optimizer = optimizer, loader = loader, model = model, loss_func = loss_function)


