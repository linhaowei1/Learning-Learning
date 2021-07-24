from __future__ import print_function
from utils import get_args
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import json
import time
import random
import nlpaug.augmenter.word as naw
import pdb
import pickle as pkl
import pdb
from tqdm import tqdm
from bert import *
from transformers import DistilBertForSequenceClassification, AdamW, BertModel


def evaluate(epoch_number):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
    
            if args.cuda:
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                labels = labels.cuda()

            pred = model.forward(inputs, labels)
            loss = criterion(pred, labels)
            total_loss += loss.data

            prediction = torch.max(pred, 1)[1]
            total_correct += torch.sum((prediction == labels).float())

        ave_loss = total_loss / (len(data_val) // args.batch_size)

    return ave_loss, total_correct.data / len(data_val)


def train(epoch_number):
    global best_val_loss, best_acc
    model.train()
    total_loss = 0
    start_time = time.time()
    batch = 0

    for inputs, labels in dataloader:
        batch += 1
        if args.cuda:
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            labels = labels.cuda()
        pred = model.forward(inputs)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'.format(
                  epoch_number, batch, len(dataloader),
                  elapsed * 1000 / args.log_interval, total_loss / args.log_interval))
            total_loss = 0
            start_time = time.time()

    evaluate_start_time = time.time()
    val_loss, acc = evaluate(epoch_number)
    best_val_loss = min(val_loss, best_val_loss)
    best_acc = max(best_acc, acc)
    print('-' * 89)
    fmt = '| evaluation | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f} '
    print(fmt.format((time.time() - evaluate_start_time), val_loss, acc))
    print("current best_val_loss = {:5.4f}, best_acc = {:8.4f}".format(best_val_loss, best_acc))
    print('-' * 89)


if __name__ == '__main__':
    writer = open('bert_acc_freeze.txt', 'a')
    with torch.cuda.device(2):
        for ep in range(14):
            args = get_args()
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                if not args.cuda:
                    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
                else:
                    torch.cuda.manual_seed(args.seed)
            random.seed(args.seed)
            best_val_loss = 10000.0
            best_acc = 0.0

            model = BERT(flag=ep)
            if args.cuda:
                model = model.cuda()

            criterion = CELoss()
            if args.cuda:
                criterion = criterion.cuda()
            
            for param in model.model.base_model.parameters():
                param.requires_grad = False

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

            data_train = SST1Dataset('./SST1/train.txt')
            dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

            data_val = SST1Dataset('./SST1/val.txt')
            val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

            for epoch in range(50):
                train(epoch)
            
            writer.write("bert_flag =" + str(ep) + ' : ' + str(best_acc) + '\n')

    writer.close()

            
