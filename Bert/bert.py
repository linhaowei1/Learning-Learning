import torch
import numpy as np
from torch import nn
from transformers import DistilBertForSequenceClassification, AdamW, BertModel, AutoTokenizer
import math
import torch.nn.functional as f
import pdb
from torch.utils.data import Dataset, DataLoader

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def sst1loader(filename):
    sentences = []
    labels = []
    lines = open(filename, 'r').readlines()
    for line in lines:
        label = int(line[0])
        text = str(line[2:])
        sentences.append(text)
        labels.append(label)
    return sentences, labels

def collate_fn(batch):
    x, y = zip(*batch)
    x = tokenizer(list(x), padding=True, truncation=True, max_length=120, return_tensors="pt")
    y = torch.LongTensor(y)
    return x, y

class SST1Dataset(Dataset):
    def __init__(self, filename):
        self._data, self._label = sst1loader(filename)
        self._len = len(self._data)
    
    def __getitem__(self, item):
        return self._data[item], self._label[item]
    
    def __len__(self):
        return self._len

class BERT(nn.Module):
    def __init__(self, convert_mode=False, flag=0):
        super(BERT, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()
        self.pre_classifier = nn.Sequential(
            nn.Linear(768, 600),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(600, 300)
        )
        self.classifier = nn.Linear(300, 5)
        self.dropout = nn.Dropout(0.1)
        self.flag = flag
        self.projection_head = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300)
        )

    def forward(self, input, label = None):
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        outputs = self.model(input_ids.long(), attention_mask=attention_mask)
        if self.flag == 0:
            sentence = outputs[0][:, 0]
            output = self.pre_classifier(sentence)
            output = nn.ReLU()(output)
            output = self.dropout(output)
            embeddings = self.classifier(output)
        elif self.flag == 1:
            sentence = torch.mean(outputs[0], 1)
            output = self.pre_classifier(sentence)
            output = nn.ReLU()(output)
            output = self.dropout(output)
            embeddings = self.classifier(output)
        elif self.flag == 2:
            sentence = torch.mean(outputs[0], 1)
            embeddings = sentence
        elif self.flag == 3:
            sentence = outputs[0][:, 0]
            embeddings = sentence
        elif self.flag == 4:
            sentence = torch.mean(outputs[0][:, 1:], 1)
            embeddings = sentence
        elif self.flag == 5:
            sentence = torch.mean(outputs[0][:, 1:], 1)
            output = self.pre_classifier(sentence)
            output = nn.ReLU()(output)
            output = self.dropout(output)
            embeddings = self.classifier(output)
        elif self.flag == 6:
            sentence = outputs[2][-2][:, 0]
            output = self.pre_classifier(sentence)
            output = nn.ReLU()(output)
            output = self.dropout(output)
            embeddings = self.classifier(output)
        elif self.flag == 7:
            sentence = outputs[2][-2][:, 0]
            output = self.pre_classifier(sentence)
            output = nn.ReLU()(output)
            output = self.dropout(output)
            embeddings = self.classifier(output)
        elif self.flag == 8:
            sentence = outputs[2][-7][:, 0]
            embeddings = sentence
        elif self.flag == 9:
            sentence = torch.mean(outputs[2][-7][:, 1:], 1)
            embeddings = sentence
        elif self.flag == 10:
            sentence = outputs[2][-10][:, 0]
            embeddings = sentence
        elif self.flag == 11:
            sentence = torch.mean(outputs[2][1][:, 1:], 1)
            embeddings = sentence
        elif self.flag == 12:
            sentence = torch.mean(outputs[2][0][:, 1:], 1)
            output = self.pre_classifier(sentence)
            output = nn.ReLU()(output)
            output = self.dropout(output)
            embeddings = self.classifier(output)
        elif self.flag == 13:
            sentence = outputs[2][1][:, 0]
            embeddings = sentence
 
        output = f.softmax(embeddings, dim=1)
        return output

    def CLforward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model(input_ids.long(), attention_mask=attention_mask)
        sentence = outputs[0][:, 0]
        output = self.dense1(sentence)
        output = nn.ReLU()(output)
        output = self.dropout(output)
        embeddings = self.dense2(output)
        return embeddings

class CLLoss(nn.Module):  # contrastive loss (simCLR)
    def __init__(self):
        super(CLLoss, self).__init__()
        self.temperature = 0.5
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, pred1, pred2):
        #pdb.set_trace()
        batch_size = pred1.size(0)
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to('cuda')

        features = torch.cat([pred1, pred2], dim = 0)
        features = f.normalize(features, dim=1)

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
        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.c_e = nn.CrossEntropyLoss()
    
    def forward(self, pred, targets):
        class_loss = self.c_e(pred, targets)
        return class_loss