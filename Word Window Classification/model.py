from dataset import *
import torch
import torch.nn as nn

class Classifier(nn.Module):
    
    def __init__(self, hyperparameters, vocab_size, pad_ix = 0):
        super(Classifier, self).__init__()

        self.window_size = hyperparameters["window_size"]
        self.embed_dim = hyperparameters["embed_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.freeze_embeddings = hyperparameters["freeze_embeddings"]
        self.vocab_size = vocab_size
        self.pad_ix = pad_ix

        self.embed_layer = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx = pad_ix)
        if self.freeze_embeddings:
            self.embed_layer.weight.requires_grad = False

        full_size_window = 2 * self.window_size + 1

        self.hidden_layer = nn.Sequential(
            nn.Linear(full_size_window * self.embed_dim, self.hidden_dim),
            nn.Tanh()
        )

        self.output_layer = nn.Linear(self.hidden_dim, 1)
        self.prob = nn.Sigmoid()
    
    def forward(self, inputs):
        """
            inputs : B X L
        """
        B, L = inputs.size()
        full_size_window = self.window_size * 2 + 1
        token_windows = inputs.unfold(1, full_size_window, 1)
        _, adjust_len, _ = token_windows.size()
        assert (B, adjust_len, full_size_window) == token_windows.size()
        # token_windows : B X Adjust X L~ 
        token_windows = self.embed_layer(token_windows)
        # token_windows : B X Adjust X L~ X D
        token_windows = token_windows.view(B, adjust_len, -1)
        token_windows = self.hidden_layer(token_windows)
        probability = self.prob(self.output_layer(token_windows)).view(B,-1)
        return probability
    


def loss_function(batch_outputs, batch_labels, batch_length):
    bceloss = nn.BCELoss()
    loss = bceloss(batch_outputs, batch_labels.float())

    loss = loss / batch_length.sum().float()

    return loss

def train_epoch(optimizer, loader, model, loss_func):
    total_loss = 0.0
    for batch_inputs, batch_labels, batch_lengths in loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_function(outputs, batch_labels, batch_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def train(epoch_num, optimizer, loader, model, loss_func):
    for epoch in range(epoch_num):
        epoch_loss = train_epoch(optimizer, loader, model, loss_func)
        if epoch % 100 == 0:
            print(f'epoch : {epoch}, loss = {epoch_loss}')
    torch.save(model.state_dict(), 'params.pkl')



vocabulary, datasets, inx2word, word2inx = make_data(corpus, LOCATION)
batch_size = 2
shuffle = True
window_size = 2

hyperparameters = {
    "window_size" : 2,
    "embed_dim" : 250,
    "hidden_dim" : 250,
    "freeze_embeddings" : False
}

vocab_size = len(word2inx)
model = Classifier(hyperparameters, vocab_size, word2inx['<pad>'])
