from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
import pdb
import math
import pickle as pkl
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, config, embedding):
        super(LSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = embedding
        self.bilstm = nn.LSTM(config['embsize'], config['hiddensize'],
                              config['layers'], dropout=config['dropout'], bidirectional=True)
        self.nlayers = config['layers']
        self.nhid = config['hiddensize']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0

    def forward(self, inp, hidden, lenth, cuda_av):
        emb = self.drop(self.encoder(inp))
        pack = nn.utils.rnn.pack_padded_sequence(emb, lenth)
        packed, hc = self.bilstm(pack, hidden)
        unpacked = nn.utils.rnn.pad_packed_sequence(packed)
        outp_all = unpacked[0]
        lenth = unpacked[1]
        if self.pooling == 'mean':
            outp = torch.mean(outp_all, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp_all, 0)[0].squeeze()
        elif self.pooling == 'last':
            outp = hc[0][1]
        return outp, emb, outp_all

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))


class Classifier(nn.Module):
    def __init__(self, config, cuda_av):  # "dropout": the dropout rate
        # "embsize": embedding size
        # "vocabnum": number of words
        # "dictionary": dic of words and ids
        # "hiddensize": the size of hidden vector of lstm
        # "layers": the number of lstm layers
        super(Classifier, self).__init__()
        self.embsize = config["embsize"]
        self.dictionary = config['dictionary']
        self.embedd = nn.Embedding(config["vocabnum"], self.embsize)
        self.maxlenth = config['maxlenth']
        #self.batch_size = config['batch-size']
        self.encoder = LSTM(config, self.embedd)
        self.fc = nn.Linear(config['hiddensize'], config['nfc'])
        self.dense = nn.Linear(config['hiddensize'], config['hiddensize'])
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.cuda_av = cuda_av

    def forward(self, inp, hidden, lenth, epoch=0):
        # predict the class
        outp, _, outp_all = self.encoder.forward(
            inp, hidden, lenth, self.cuda_av)
        outp = outp.view(outp.size(0), -1)
        out = self.drop(self.dense(self.relu(self.dense(outp))))
        return out
    
    def get_feature(self, inp, hidden, lenth, epoch=0):
        # predict the class
        outp, _, outp_all = self.encoder.forward(
            inp, hidden, lenth, self.cuda_av)
        outp = outp.view(outp.size(0), -1)
        return outp

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]

    def init_emb_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)
        self.w1.weight.data.uniform_(-init_range, init_range)
        self.w2.weight.data.uniform_(-init_range, init_range)
        self.w3.weight.data.uniform_(-init_range, init_range)
        self.w4.weight.data.uniform_(-init_range, init_range)
        self.embedd.weight.data.uniform_(-init_range, init_range)


class myLoss(nn.Module):  # contrastive loss (simCLR)
    def __init__(self):
        super(myLoss, self).__init__()
        self.temperature = 0.5
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, pred1, pred2):
        #pdb.set_trace()
        batch_size = pred1.size(0)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to('cuda')

        features = torch.cat([pred1, pred2], dim = 0)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.long).to('cuda')
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0]).to('cuda')

        logits = logits / self.temperature
        loss = self.criterion(logits, labels.long())
        return loss/batch_size
