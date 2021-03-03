import torch
import numpy as np
from torch import nn
from transformers import DistilBertForSequenceClassification, AdamW, BertModel
import math
import torch.nn.functional as f
import pdb

class BERT(nn.Module):
    def __init__(self, config, convert_mode=False, flag=0):
        super(BERT, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()
        self.pre_classifier = nn.Linear(768, 300)
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
 
        output = f.softmax(embeddings)
        return output