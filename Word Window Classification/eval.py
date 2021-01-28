from model import *
from torch.utils.data import DataLoader
from functools import partial

vocabulary, datasets, inx2word, word2inx = make_data(corpus, LOCATION)
test_corpus = [
    'I am born in hebei',
    'I make a trip to hongkong',
    "she comes from paris",
    "Anxi and xiamen are beautiful",
    "you can come to yunnan",
    "America is a beautiful place",
    "Not everyone loves England",
    "I and Lili went to Europe"
]
test_sentences = [s.lower().split() for s in test_corpus]
test_labels = [
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1]
]

# Create a test loader
model.load_state_dict(torch.load('params.pkl'))

test_data = list(zip(test_sentences, test_labels))
batch_size = 1
shuffle = False
window_size = 2
collate_fn = partial(custom_collate_fn, window_size=2, word2inx=word2inx)
test_loader = torch.utils.data.DataLoader(test_data, 
                                           batch_size=1, 
                                           shuffle=False, 
                                           collate_fn=collate_fn)
model.freeze_embeddings = True
for test_instance, labels, _ in test_loader:
    outputs = model.forward(test_instance)
    print(labels)
    outputs = [1 if item > 0.5 else 0 for item in outputs[0]]
    print(outputs)