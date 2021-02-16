from models import *
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
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
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 


def package(data, volatile=False):
    """Package data for training / evaluation."""
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

def evaluate():
    TARGET = []
    PROB = []
    model.eval()  # turn on the eval() switch to disable dropout

    for batch, i in enumerate(range(0, len(data_val), 1)):
        data, targets, lenth = package(data_val[i:i+1], volatile=False)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        #pdb.set_trace()
        with torch.no_grad():
            hidden = model.init_hidden(data.size(1))
            pred = model.get_feature(data, hidden, lenth, 1)
            score = (feature * pred).sum(dim = 1).max().item()
            TARGET.append(targets.item())
            PROB.append(score)
            print("score = ", score, "target = ", targets.item())

    fpr, tpr, thresholds = metrics.roc_curve(TARGET, PROB, pos_label=4)
    thresh = thresholds[np.argmax(tpr - fpr)]
    print("thershold = ", thresh)
    acc = 0
    for i in range(len(TARGET)):
        if PROB[i] >= thresh and TARGET[i] == 4:
            acc += 1
        elif PROB[i] < thresh and TARGET[i] != 4:
            acc += 1
    acc /= len(TARGET)
    print("acc=", acc)
    print("finally, auc = {}".format(metrics.auc(fpr, tpr)))
    print("PROB example = ", PROB[20:40])
    print("TARGET example = ", TARGET[20:40])


        
def deal_train(train_data):
    new_train_data = []
    for data in train_data:
        #data['text'] = aug(data['text'])
        if data['label'] == 4:
            data['text'] = aug(data['text'])
            new_train_data.append(data)
    return new_train_data

def deal_val(val_data):
    new_val_data = []
    for data in val_data:
        '''
        item1 = SwapTrans(data['text'])
        data_t = dict(data)
        data_t['text'] = item1
        data_t['label'] = 1
        data['text'] = aug(data['text'])
        data['label'] = 4'''
        data['text'] = aug(data['text'])
        new_val_data.append(data)
        #new_val_data.append(data_t)
        #pdb.set_trace()
    return new_val_data

def train():
    model.eval()
    feature = []

    for batch, i in enumerate(range(0, len(data_train), args.batch_size)):
        data, targets, lenth = package(data_train[i:i+args.batch_size], volatile=False)
        if args.cuda:
            data = data.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            hidden = model.init_hidden(data.size(1))
            feats = model.get_feature(data, hidden,lenth, 1)
            feats = F.normalize(feats, dim=1)
            feature += feats.chunk(args.batch_size, dim = 0)
    
    return torch.cat(tuple(feature), 0).cuda()

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
    model.load_state_dict(torch.load('params_csi_transform=swap0.6_pos=4.pkl'))  

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
    data_train = deal_train(data_train)
    data_val = open(args.val_data).readlines()
    data_val = list(map(lambda x: json.loads(x), data_val))
    data_val = deal_val(data_val)

    feature = train()
    evaluate()

    
