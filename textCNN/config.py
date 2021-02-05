import preprocess as P
import model as M
from torch.utils.data import DataLoader
from functools import partial
import torch
import torch.nn as nn

train_root = 'text_classification_data/train.txt'
dev_root = 'text_classification_data/dev.txt'
test_root = 'text_classification_data/test.txt'

hyperparameters = {
    'embed_dim' : 300,
    'sent_len' : 15,
    'in_channel' : 1,
    'out_channel' : 100,
    'ker_size1' : 3,
    'ker_size2' : 4,
    'ker_size3' : 5,
    'drop_rate' : 0.5,
    'class_num' : 4,
    'weight_decay' : 1e-2,
    'hidden_dim' : 300,
    'batch_size' : 128,
    'lr' : 1e-3
}

Stopwords = P.stopwords("stopwords.txt")

train_data = P.init_data(train_root, Stopwords, hyperparameters['sent_len'])
dev_data = P.init_data(dev_root, Stopwords, hyperparameters['sent_len'])
test_data = P.init_data(test_root, Stopwords, hyperparameters['sent_len'])

vocab = P.vocabulary(train_data, dev_data, test_data)
idx2word, word2idx = P.word2idx(vocab)

batch_train = P.make_batch(train_data, word2idx)
batch_dev = P.make_batch(dev_data, word2idx)
batch_test = P.make_batch(test_data, word2idx)

collate_fn = partial(P.custom_collate_fn, word2idx=word2idx, device = 'cpu')
train_loader = DataLoader(batch_train, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(batch_dev, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(batch_test, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)

model = M.textCNN(hyperparameters, len(vocab), padding_idx = word2idx['<pad>'])

# you can load the trained params here
model.load_state_dict(torch.load('model.pkl'))

optimizer = torch.optim.Adam(model.parameters(), lr = hyperparameters['lr'], weight_decay = hyperparameters['weight_decay'])

Loss = nn.CrossEntropyLoss()

best_acc = 0

epoch_num = 20