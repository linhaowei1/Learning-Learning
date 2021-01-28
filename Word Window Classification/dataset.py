import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
corpus = [
    ['China','and','Beijing','are','beautiful'],
    ['I','come','from','Fujian'],
    ['Hello','You','are','really','like','a','Guangdong','person'],
    ['You','should','go','to','Beijing','for','vacation'],
    ['Yestoday','she','went','to','Taiwan'],
    ['Yestoday','I','go','to','Shandong'],
    ['You','should','come','to','Yunnan'],
    ['No','one','likes','Beijing'],
    ['Everyone','likes','Jiangxi'],
    ['Fujian','is','a','good','place'],
    ['Tianjin','is','really','hot','yestoday'],
    ['Hainan','is','the','hottest','place','in','China'],
    ['Fujian','lies','in','the','east','of','China'],
    ['Taiwan','is','near','Fujian'],
    ['She','comes','from','Fujian','Xiamen']
]
LOCATION = ['Fujian', 'Guangdong', 'Beijing','Taiwan','Shandong','Yunnan','Beijing','Jiangxi','Fujian',
'Tianjin', 'Hainan','China','Xiamen']

def make_data(corpus, LOCATION):
    corpus = [[word.lower() for word in sen] for sen in corpus]
    LOCATION = [word.lower() for word in LOCATION]
    label = [[1 if word in LOCATION else 0 for word in sen] for sen in corpus]
    vocabulary = set(w.lower() for s in corpus for w in s)
    vocabulary.add('<unk>')
    vocabulary.add('<pad>')

    data = [sen for sen in corpus]
    dataset = list(zip(data, label))
    
    inx2word = sorted(list(vocabulary))
    word2inx = {word:inx for inx,word in enumerate(inx2word)}

    return vocabulary, dataset, inx2word, word2inx

def custom_collate_fn(dataset, window_size, word2inx):
    data,label = zip(*dataset)
    length = torch.LongTensor([len(sen) for sen in data])
    data_pad = ['<pad>'] * window_size
    data = [data_pad + sen + data_pad for sen in data]
    indices = [[word2inx.get(token, word2inx['<unk>']) for token in sen] for sen in data]
    indices = [torch.LongTensor(x) for x in indices]
    indices = nn.utils.rnn.pad_sequence(indices, batch_first = True, padding_value = word2inx['<pad>'])

    label = [torch.LongTensor(x) for x in label]
    label = nn.utils.rnn.pad_sequence(label, batch_first = True, padding_value = 0)

    return indices, label, length
