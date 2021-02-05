import torch
import torch.nn as nn
import pdb

class textCNN(nn.Module):
    def __init__(self, hyperparameters, vocab_size, padding_idx):
        super(textCNN, self).__init__()
        self.embed_dim = hyperparameters['embed_dim']
        self.sent_len = hyperparameters['sent_len']
        self.in_channel = hyperparameters['in_channel']
        self.out_channel = hyperparameters['out_channel']
        self.ker_size1 = (hyperparameters['ker_size1'], self.embed_dim)
        self.ker_size2 = (hyperparameters['ker_size2'], self.embed_dim)
        self.ker_size3 = (hyperparameters['ker_size3'], self.embed_dim)
        self.drop_rate = hyperparameters['drop_rate']
        self.class_num = hyperparameters['class_num']
        self.hidden_dim = hyperparameters['hidden_dim']

        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx = padding_idx)
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, self.ker_size1)
        self.conv2 = nn.Conv2d(self.in_channel, self.out_channel, self.ker_size2)
        self.conv3 = nn.Conv2d(self.in_channel, self.out_channel, self.ker_size3)
        self.maxpool1 = nn.MaxPool1d(self.sent_len-self.ker_size1[0]+1)
        self.maxpool2 = nn.MaxPool1d(self.sent_len-self.ker_size2[0]+1)
        self.maxpool3 = nn.MaxPool1d(self.sent_len-self.ker_size3[0]+1)
        self.dropout = nn.Dropout(self.drop_rate)
        self.dense1 = nn.Linear(self.out_channel * 3, self.hidden_dim, bias = True)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.hidden_dim, self.class_num, bias = True)
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, input):
        # input = (batch_size, sent_len)
        embed = self.embedding(input)
        #pdb.set_trace()
        # embed = (batch_size, sent_len, embed_dim)
        embed = embed.unsqueeze(1) # add 1 dimension -> 1 channel
        # embed = (batch_size, in_channel, sent_len, embed_dim)
        conv1 = self.conv1(embed).squeeze(-1)
        conv2 = self.conv2(embed).squeeze(-1)
        conv3 = self.conv3(embed).squeeze(-1)
        # conv = (batch_size, out_channel, sent_len - ker_size + 1)
        pool1 = self.maxpool1(conv1).squeeze(-1)
        pool2 = self.maxpool2(conv2).squeeze(-1)
        pool3 = self.maxpool3(conv3).squeeze(-1)
        # pool = (batchsize, out_channel, 1)
        output = torch.cat((pool1,pool2,pool3), dim = -1)
        # output = (batchsize, out_channel * 3)
        output = self.dense1(self.dropout(output))
        # output = (batchsize, hidden_size)
        output = self.relu(output)
        pred_all = self.softmax(self.dense2(output))
        # output = (batchsize, class_num)
        pred = torch.argmax(pred_all, dim = -1)
        return pred_all, pred







