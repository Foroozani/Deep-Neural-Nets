import os
import re
import sys
import pickle
from glob import glob
from tqdm import tqdm_notebook

import nltk
import spacy
import logging

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchtext
import torchtext.datasets as datasets
from torchtext import data, vocab

from IPython.display import display

LOGGER = logging.getLogger("toxic_dataset")

def prepare_csv(train_csv, test_csv, split=0.2, seed=999):
    if not os.path.exists('data'):
        os.mkdir('data')
        
    # read train csv file
    df_train = pd.read_csv(train_csv)
    df_train["comment_text"] = df_train.comment_text.str.replace("\n", " ")
    
    # create validation data
    idx = np.arange(df_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * split)
    df_train.iloc[idx[val_size:], :].to_csv("data/dataset_train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv("data/dataset_val.csv", index=False)
    
    # read test csv file
    df_test = pd.read_csv(test_csv)
    df_test["comment_text"] = df_test.comment_text.str.replace("\n", " ")
    df_test.to_csv("data/dataset_test.csv", index=False)


#Toxic Comments Dataset

data_dir = 'D:/datasets/kaggle/toxic_comments'

train_csv = f'{data_dir}/train.csv'
test_csv = f'{data_dir}/test.csv'

# batch_size = 2

train_df = pd.read_csv(train_csv)
display(train_df.sample(n=5))

sos_token = 0
eos_token = 1

class Vocabulary(object):
    def __init__(self):
        self.word2index = {"<sos>": 0, "<eos>": 1}
        self.word2count = {}
        self.index2word = {}
        self.count = 2
    
    def add_word(self, word):
        if not word in self.word2index:
            self.word2index[word] = self.count
            self.word2count[word] = 1
            self.index2word[self.count] = word
            self.count += 1
        else:
            self.word2count[word] += 1
    
    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)
            
    def __len__(self):
        return self.count

vocab = Vocabulary()
all_comments_text = train_df["comment_text"]
for text in tqdm_notebook(all_comments_text, desc='Building vocabulary'):
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        vocab.add_sentence(sent)
        
with open('vocab.pkl', 'bw') as f:
    pickle.dump(vocab, f)

vocab = pickle.load(open('vocab.pkl', 'rb'))

print(len(vocab))

print("All:", len(train_df))
print("toxic:", len(train_df[train_df['toxic'] == 1]))
print("severe_toxic:", len(train_df[train_df['severe_toxic'] == 1]))
print("obscene:", len(train_df[train_df['obscene'] == 1]))
print("threat:", len(train_df[train_df['threat'] == 1]))
print("insult:", len(train_df[train_df['insult'] == 1]))
print("identity_hate:", len(train_df[train_df['identity_hate'] == 1]))


NLP = spacy.load('en')
MAX_CHARS = 20000

def tokenizer(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", 
        str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [x.text for x in NLP.tokenizer(comment) if x.text != " "]

def get_dataset(train_scv, test_csv, split=0.2, fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only supports all lower cases
        lower = True
    
    LOGGER.debug("Preparing CSV files...")
#     prepare_csv(train_csv, test_csv, split)
    
    comment = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        tensor_type=torch.cuda.LongTensor,
        lower=lower
    )
    
    print("Reading train csv file...")
    train, val = data.TabularDataset.splits(
        path='data/', format='csv', skip_header=True,
        train='dataset_train.csv', validation='dataset_val.csv',
        fields=[
            ('id', None),
            ('comment_text', comment),
            ('toxic', data.Field(
                use_vocab=False, sequential=False,
                tensor_type=torch.cuda.ByteTensor)),
            ('severe_toxic', data.Field(
                use_vocab=False, sequential=False, 
                tensor_type=torch.cuda.ByteTensor)),
            ('obscene', data.Field(
                use_vocab=False, sequential=False, 
                tensor_type=torch.cuda.ByteTensor)),
            ('threat', data.Field(
                use_vocab=False, sequential=False, 
                tensor_type=torch.cuda.ByteTensor)),
            ('insult', data.Field(
                use_vocab=False, sequential=False, 
                tensor_type=torch.cuda.ByteTensor)),
            ('identity_hate', data.Field(
                use_vocab=False, sequential=False, 
                tensor_type=torch.cuda.ByteTensor)),
        ])
    
    print("Reading test csv file...")
    test = data.TabularDataset(
        path='data/dataset_test.csv', format='csv', 
        skip_header=True,
        fields=[
            ('id', None),
            ('comment_text', comment)
        ])
    
    print("Building vocabulary...")
    comment.build_vocab(
        train, val, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    
    print("Done preparing the datasets")
    return train, val, test
# time
train_ds, valid_ds, test_ds = get_dataset(train_csv, test_csv, split=0.2)


print(len(train_ds.examples))
print(len(valid_ds.examples))
print(len(test_ds.examples))

train_ds.fields

def get_iterator(dataset, batch_size, train=True, 
    shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=0,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    return dataset_iter

batch_size = 1
train_iter = get_iterator(train_ds, batch_size, train=True, shuffle=True, repeat=True)

for i, examples in enumerate(train_iter):
    x = examples.comment_text # (fix_length, batch_size) Tensor
    y = torch.stack([
        examples.toxic, 
        examples.severe_toxic, 
        examples.obscene,
        examples.threat, 
        examples.insult, 
        examples.identity_hate
    ], dim=1)
    
    print(x)
    print(y)
    if i >= 1: break

# Loading pre-trained word vectors

#Encoder RNN
use_gpu = torch.cuda.is_available()

def to_var(x, volatile=False):
    x = Variable(x, volatile=volatile)
    return x.cuda() if use_gpu else x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=6, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hidden):
        output = self.embedding(x)
        for i in range(self.num_layers):
            output, hidden = self.gru(output, hidden)
        output = self.out(hidden)
        output = F.relu(output)
        output = F.dropout(output, p=0.1)
        output = F.sigmoid(output)
        return output
    
    def init_hidden(self):
        return to_var(torch.zeros((1, 1, self.hidden_size)))

hidden_size = 128
num_layers = 1
vocab = train_ds.fields['comment_text'].vocab
print(len(vocab))
model = EncoderRNN(len(vocab), hidden_size, num_classes=6, num_layers=1)

if use_gpu:
    model = model.cuda()

print(model)

criterion = nn.BCELoss()
if use_gpu:
    criterion = criterion.cuda()
    
optimizer = optim.Adam(rnn.parameters(), lr=0.002)

num_epochs = 1


for epoch in range(num_epochs):
    epoch_loss = 0.0
#     h = to_var(torch.zeros((num_layers, batch_size, hidden_size)))
    h = rnn.init_hidden()
    
    for i, examples in tqdm_notebook(enumerate(train_iter)):
        x = examples.comment_text
        y = torch.stack([examples.toxic, examples.severe_toxic, examples.obscene,
                         examples.threat, examples.insult, examples.identity_hate], dim=1)
        
        # forward step
        output = rnn(x, h)
        
        # loss
        loss = criterion(output, y.float().view(1, 1, -1))
        
        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # stats
        sys.stdout.flush()
        sys.stdout.write('\r loss = {:.5f}'.format(loss.data[0]))
        
        if i > len(train_ds.examples): break

# LSTM Classifier

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = to_var(torch.zeros(1, self.batch_size, self.hidden_dim))
        c0 = to_var(torch.zeros(1, self.batch_size, self.hidden_dim))
        return h0, c0

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        output, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(output[-1])
        return y

## parameter setting
epochs = 50
batch_size = 5
learning_rate = 0.002

embedding_dim = 100
hidden_dim = 50
seq_len = 100
num_classes = 6

model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim, 
                       vocab_size=len(vocab), label_size=num_classes, 
                       batch_size=batch_size)

if use_gpu:
    model = model.cuda()

criterion = nn.BCELoss()
if use_gpu:
    criterion = criterion.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_loss_ = []
test_loss_ = []
train_acc_ = []
test_acc_ = []

for epoch in range(num_epochs):
#     optimizer = adjust_learning_rate(optimizer, epoch)

    ## training epoch
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    
    for i, traindata in enumerate(train_loader):
        train_inputs, train_labels = traindata
        train_labels = torch.squeeze(train_labels)

        if use_gpu:
            train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
        else: train_inputs = Variable(train_inputs)

        model.zero_grad()
        model.batch_size = len(train_labels)
        model.hidden = model.init_hidden()
        output = model(train_inputs.t())

        loss = loss_function(output, Variable(train_labels))
        loss.backward()
        optimizer.step()

        # calc training acc
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == train_labels).sum()
        total += len(train_labels)
        total_loss += loss.data[0]

    train_loss_.append(total_loss / total)
    train_acc_.append(total_acc / total)
    ## testing epoch
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for iter, testdata in enumerate(test_loader):
        test_inputs, test_labels = testdata
        test_labels = torch.squeeze(test_labels)

        if use_gpu:
            test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
        else: test_inputs = Variable(test_inputs)

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs.t())

        loss = loss_function(output, Variable(test_labels))

        # calc testing acc
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.data[0]
    test_loss_.append(total_loss / total)
    test_acc_.append(total_acc / total)

    print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
          % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))





