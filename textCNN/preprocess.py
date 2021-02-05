import jieba
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

# the stopwords #
def stopwords(root):
    f = open(root, "r", encoding = 'utf8')
    lines = f.readlines()
    stopwords = [word.strip() for word in lines]
    stopwords.append(' ')
    f.close()
    return stopwords

# the training data #

def init_data(root, stopwords, maxlen):
    f = open(root, 'r', encoding = 'utf8')
    lines = f.readlines()
    dataset = []
    pad_lst = ['<pad>']
    for line in lines:
        packed = line.strip().split("\t")
        packed[0] = list(jieba.cut(packed[0]))
        packed[0] = [word for word in packed[0] if word not in stopwords and ' ' not in word]
        if len(packed[0]) < maxlen:
            packed[0] += pad_lst * (maxlen - len(packed[0]))
        else:
            packed[0] = packed[0][:maxlen]
        dataset.append({'text':packed[0], 'label':packed[1]})
    return dataset


# vocab #
def vocabulary(train_data, dev_data, test_data):
    vocab = [word for data in train_data for word in data['text']]
    vocab += [word for data in dev_data for word in data['text']]
    vocab += [word for data in test_data for word in data['text']]
    vocab.append('<pad>')
    return set(vocab)


def word2idx(vocab):
    idx2word = sorted(list(vocab))
    word2idx = {}
    for num, word in enumerate(idx2word):
        word2idx[word] = num
    return idx2word, word2idx


def make_batch(data, word2idx):
    train_sentences = []
    train_labels = []
    for item in data:
        item_text = [word2idx[word] for word in item['text']]
        train_sentences.append(item_text)
        train_labels.append(eval(item['label']))
    return list(zip(train_sentences, train_labels))


def custom_collate_fn(batch, word2idx, device):
    x, y = zip(*batch)

    pad_token_ix = word2idx['<pad>']
    x = [torch.LongTensor(x_i, device = device) for x_i in x]
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first = True, padding_value = pad_token_ix)
    y = torch.LongTensor(y, device = device)

    return x_padded, y

