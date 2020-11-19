import pandas as pd 
import numpy as np
import torch
import jieba

## --- process data --- ##
train_path = 'data/train.txt'
dev_path = 'data/dev.txt'
test_path = 'data/test.txt'
stop_path = 'data/cn_stopwords.txt'

stopwords = pd.read_csv('data/cn_stopwords.txt',index_col = False, quoting=3, sep='\t', names=['stopword'],encoding='utf-8')
stopwords = stopwords['stopword'].values

def make_data(path):
    dataset = pd.read_table(path,encoding='utf-8',header=None,index_col=False,sep='\t')
    sentence = []
    for line,catagory in dataset.values:
        segs = jieba.lcut(line)
        segs = filter(lambda x:len(x)>1, segs)
        segs = filter(lambda x:x not in stopwords, segs)
        sentence.append((" ".join(segs),catagory))
    return sentence

train_set = make_data(train_path)
dev_set = make_data(dev_path)
test_path = make_data(test_path)

## -- model -- ##

MAX_LENGTH = 100
MIN_LENGTH = 2
EMBEDDING_SIZE = 20
N_FILTERS = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
