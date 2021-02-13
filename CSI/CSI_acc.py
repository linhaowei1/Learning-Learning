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

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 


def test_pretrain_acc(pred, targets, size):
    cos = torch.nn.CosineSimilarity(dim = 0)
    tot = 0
    correct = 0
    for i in range(32):
        # 遍历每个label编号
        pair = [] #记录相同的pair
        Index = []
        tot1 = 0.0
        for index in range(len(targets)): #遍历下标
            if targets[index] == i: #如果该下标所在元素的target和i相同
                pair.append(pred[index]) #放进pair
                Index.append(index)
            if len(pair) == 2:
                break
        frac = cos(pair[0],pair[1])
        maxi = 0
        for index in range(len(targets)): #遍历下标
            if index == Index[0]:
                continue
            maxi = max(cos(pair[0],pred[index]), maxi)
        if maxi == frac:
            correct += 1
        tot+=1
    return correct/tot

def package(data, volatile=False):
    """Package data for training / evaluation."""
    # data = sorted(data, key = lambda x: len(x['text']), reverse=True)
    data = sorted(data, key = lambda x: len(x['text']), reverse=True)
    dat = map(lambda x: list(map(lambda y: dictionary.word2idx.get(y, 0), x['text'])), data)
    dat = list(dat)
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = list(map(lambda x: x['label'], data))
    lenth = list(map(lambda x: len(x['text']), data))
    #targets = list(targets)
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
corpus = []
cos = torch.nn.CosineSimilarity(dim = 1)
def sim(z1,z2):
    return cos(z1,z2).view(-1)

def evaluate(epoch_number):
    total_correct = 0
    total_data = 0
    total_1 = 0
    total_0 = 0
    correct_1 = 0
    correct_0 = 0
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    
    for batch, i in enumerate(range(0, len(data_val), 1)):
        data, targets, lenth = package(data_val[i:i+1], volatile=False)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        #pdb.set_trace()
        hidden = model.init_hidden(data.size(1))
        pred = model.get_feature(data, hidden,lenth, epoch_number)
        #pdb.set_trace()
        similarity = -1
        true_target = targets.item()
        tmp_label = 0
        for item in corpus:
            tmp_loss = sim(item['text'], pred).item()
            if tmp_loss > similarity:
                similarity = tmp_loss
                tmp_label = item['label'].item()
        print("similarity = {}, tmp_label = {}, target_label = {}".format(similarity,tmp_label,true_target))
        if tmp_label == true_target:
            total_correct += 1
            if tmp_label == 1:
                correct_1 += 1
            else:
                correct_0 += 1
        total_data += 1
        if true_target == 1:
            total_1 += 1
        else:
            total_0 += 1
        assert total_0 + total_1 == total_data
        if total_0 != 0 and total_1 != 0:
            print("current : ", batch, "of", len(data_val), "acc = ", total_correct/total_data, "prec1 = ", correct_0/total_0, "prec2 = ", correct_1/total_1)


def deal_val(val_data):
    new_val_data = []
    for data in val_data:
        if data['label'] == 4:
            new_val_data.append(data)
        else:
            data['label'] = 1
            new_val_data.append(data)
    return new_val_data

def deal_train(train_data):
    new_train_data = []
    for data in train_data:
        if data['label'] == 4:
            new_train_data.append(data)
            new_data = SwapTrans(data['text'])
            tmp = dict(data)
            tmp['text'] = new_data
            tmp['label'] = 1
            new_train_data.append(tmp)
            #pdb.set_trace()
    return new_train_data

def train(epoch_number):
    model.eval()

    for batch, i in enumerate(range(0, int(len(data_train)/2), 1)):
        # batch_size = N, use data from i to i+args.batch_size -1 . generate to 2N:
        data, targets, lenth = package(data_train[i:i+1], volatile=False)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        #pdb.set_trace()
        hidden = model.init_hidden(data.size(1))
        pred = model.get_feature(data, hidden,lenth, epoch_number)
        #pdb.set_trace()
        #print(targets)
        corpus.append({'text':pred,'label':targets})
        if batch % 2 == 0:
            print("current : batch = ", batch, "total = ", int(len(data_train)/2))
        #model_object.load_state_dict(torch.load('params.pkl'))

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

    # Load Dictionary
    
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
    model.load_state_dict(torch.load('params_csi_transform=swap_pos=4.pkl'))  

    if args.cuda:
        model = model.cuda()

    print(args)

    criterion = myLoss()
    if args.cuda:
        criterion = criterion.cuda()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-4, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')
    print('Begin to load data.')
    #data_train = sst2loader('train')
    data_train = open(args.train_data).readlines()
    data_train = list(map(lambda x: json.loads(x), data_train))
    data_train = deal_train(data_train)
    #pdb.set_trace()
    data_val = open(args.val_data).readlines()
    data_val = list(map(lambda x: json.loads(x), data_val))
    data_val = deal_val(data_val)
    #global name
    #name = str(args.seed)+str(args.pre)+'-'+str(args.name)+'-'+str(time.time())
    #pdb.set_trace()
    train(1)
    evaluate(1)
    