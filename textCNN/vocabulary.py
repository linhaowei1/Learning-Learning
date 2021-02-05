
train_data = init_data(train_root, stopwords)
def vocabulary(train_data):
    vocab = [word for data in train_data for word in data['text']]
    return set(vocab)
print(vocabulary(train_data))
