from __future__ import print_function
from models import *
from utils import Dictionary, get_args
from transform import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import json
import time
import random
import os
import nlpaug.augmenter.word as naw
import pdb
import pickle as pkl
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

def package(data, volatile=False):
    """Package data for training / evaluation."""
    #data = sorted(data, key = lambda x: len(x['text']), reverse=True)
    dat = map(lambda x: list(map(lambda y: dictionary.word2idx.get(y, 0), x['text'])), data)
    dat = list(dat)
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = list(map(lambda x: x['label'], data))
    lenth = list(map(lambda x: len(x['text']), data))
    maxlen = min(maxlen, 500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    dat = Variable(torch.LongTensor(dat), volatile=volatile)
    targets = Variable(torch.LongTensor(targets), volatile=volatile)
    lenth = Variable( torch.LongTensor(lenth), volatile = volatile)
    return dat.t(), targets, lenth

def deal_train(train_data, src_label = 4, trans = SwapTrans, aug = aug):
    new_train_data = []
    for data in train_data:
        if data['label'] == src_label:
            new_train_data.append(data)
            new_data = trans(data['text'])
            tmp = dict(data)
            tmp['text'] = new_data
            new_train_data.append(tmp)
    new_train_data = sorted(new_train_data, key = lambda x: len(x['text']), reverse=True)
    data_1 = []
    data_2 = []
    for item in new_train_data:
        item1 = aug(item['text'])
        item11 = dict(item)
        item11['text'] = item1
        item2 = aug(item['text'])
        item22 = dict(item)
        item22['text'] = item2
        maxlen = max(len(item1),len(item2))
        item11['text'] += ['<pad>'] * (maxlen - len(item1))
        item22['text'] += ['<pad>'] * (maxlen - len(item2))
        data_1.append(item11)
        data_2.append(item22)
    #pdb.set_trace()
    data_1 = sorted(data_1, key = lambda x: len(x['text']), reverse=True)
    data_2 = sorted(data_2, key = lambda x: len(x['text']), reverse=True)

    return data_1, data_2

def train(epoch_number):
    global best_val_loss, best_acc
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch, i in enumerate(range(0, len(data_train_1), args.batch_size)):

        data_1, _, lenth_1 = package(data_train_1[i:i+args.batch_size], volatile=False)
        data_2, _, lenth_2 = package(data_train_2[i:i+args.batch_size], volatile=False)
    
        if args.cuda:
            data_1 = data_1.cuda()
            data_2 = data_2.cuda()

        hidden = model.init_hidden(data_1.size(1))

        # get the representations of augmented data
        pred_1 = model.forward(data_1, hidden, lenth_1, epoch_number)
        pred_2 = model.forward(data_2, hidden, lenth_2, epoch_number)

        loss = criterion(pred_1,pred_2)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.data
        if batch % args.log_interval == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'.format(
                  epoch_number, batch, len(data_train_1) // args.batch_size,
                  elapsed * 1000 / args.log_interval, total_loss / args.log_interval))
            total_loss = 0
            start_time = time.time()
        torch.save(model.state_dict(), 'params_csi_transform=swap0.2_pos=4.pkl')


if __name__ == '__main__':
    # parse the arguments
    args = get_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)
    dictionary.add_word('<mask>')
    best_val_loss = None
    best_acc = None

    vocabnum = len(dictionary)
    if args.cuda:
        c = torch.ones(1)
    else:
        c = torch.zeros(1)

    print("vocabnum:", vocabnum)
    model = Classifier({
        'dropout': args.dropout,
        'vocabnum': vocabnum,
        'layers': args.layers,
        'hiddensize': args.hiddensize,
        'embsize': args.emsize,
        'pooling': 'last',
        'nfc': args.nfc,
        'dictionary': dictionary,
        'word-vector': args.word_vector,
        'class-number': args.class_number,
        'pre':args.pre,
        'maxlenth':args.maxlenth
    }, c)
    if args.cuda:
        model = model.cuda()

    print(args)
    criterion = myLoss()
    if args.cuda:
        criterion = criterion.cuda()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    print('Begin to load data.')

    data_train = open(args.train_data).readlines()
    data_train = list(map(lambda x: json.loads(x), data_train))
    data_train_1, data_train_2 = deal_train(data_train)

    for epoch in range(args.epochs):
        train(epoch)