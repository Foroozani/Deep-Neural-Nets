"""
How To Represent Words

Two main category of word representations:

    Discrete representation (e.g., one-hot encoding, Bag of Words)
    Continious representation or Word Vectors (e.g., CBoW)

"""


import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import *

# debug
from IPython.core.debugger import Pdb

# setup
use_gpu = torch.cuda.is_available()
pdb = Pdb()

"""
Vocabulary

    All the words in the corpus
    The index for each word
    The frequency of each word

"""
UNK = 0

class Vocabulary(object):
    
    def __init__(self):
        self.word2index = {}
        self.index2word = {0: '<؟>'}
        self.word2count = {}
        self.num_words = 1
        
    def add_word(self, word):
        if not word in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
        
    def add_words(self, words):
        for word in words:
            self.add_word(word)
            
    def __len__(self):
        return self.num_words

class LanguageModelDataset(Dataset):
    def __init__(self, corpus_path, split='train', vocab=None, seq_len=30, min_count=1):
        assert split in ['train', 'valid']
        self.split = split
        self.seq_len = seq_len
        self.min_count = min_count
        self.vocabulary = None
                        
        # FIRST PASS: build vocab
        if split == 'train':
            if vocab is None:
                print('Building vocabulary ...')
                self.vocabulary = Vocabulary()
                num_tokens = 0
                with open(corpus_path, encoding='utf8') as f:
                    for line in f:
                        tokens = line.split(' ') + ['<EOS>']
                        self.vocabulary.add_words(tokens)
                        num_tokens += len(tokens)
                print('Vocabulary size = {}'.format(len(self.vocabulary)))
            else:
                self.vocabulary = vocab
        else:
            self.vocabulary = vocab

        # SECOND PASS: tokenizing corpus
#         assert vocab not is None, "Vocabulary must be given!"
        print('Tokenizing corpus ...')
        self.ids = torch.LongTensor(num_tokens)
        token_idx = 0
        with open(corpus_path, encoding='utf8') as f:
            for line in f:
                tokens = line.split(' ') + ['<EOS>']
                for token in tokens:
                    if self.vocabulary.word2count[token] < min_count:
                        self.ids[token_idx] = UNK  # replace rare words with 'unk' token 
                    else: 
                        self.ids[token_idx] = self.vocabulary.word2index[token]
                    token_idx += 1

        print('Corpus size = {}'.format(num_tokens))
        
    def __get_item__(self, index):
        inputs  = self.ids[index: index + self.seq_len]
        targets = self.ids[index + 1: index + 1 + self.seq_len]
        return inputs, targets
    
    
    def __len__(self):
        return self.ids.size(0) // self.seq_len

class LanguageModelDataset(Dataset):
    def __init__(self, corpus_path, seq_len=30, min_count=1):
        self.seq_len = seq_len
        self.min_count = min_count
                        
        # FIRST PASS: build vocab
        print('Building vocabulary ...')
        self.vocabulary = Vocabulary()
        num_tokens = 0
        with open(corpus_path, encoding='utf8') as f:
            for line in f:
                tokens = line.split(' ') + ['<EOS>']
                self.vocabulary.add_words(tokens)
                num_tokens += len(tokens)
        print('Vocabulary size = {}'.format(len(self.vocabulary)))

        # SECOND PASS: tokenizing corpus
        print('Tokenizing corpus ...')
        self.ids = torch.LongTensor(num_tokens)
        token_idx = 0
        with open(corpus_path, encoding='utf8') as f:
            for line in f:
                tokens = line.split(' ') + ['<EOS>']
                for token in tokens:
                    if self.vocabulary.word2count[token] < min_count:
                        self.ids[token_idx] = UNK  # replace rare words with 'unk' token 
                    else: 
                        self.ids[token_idx] = self.vocabulary.word2index[token]
                    token_idx += 1
        print('Corpus size = {}'.format(num_tokens))
        
    def __getitem__(self, index):
        inputs  = self.ids[index: index + self.seq_len]
        targets = self.ids[index + 1: index + 1 + self.seq_len]
        return inputs, targets
    
    def __len__(self):
        return self.ids.size(0) // self.seq_len

seq_len = 30
batch_size = 20
min_count = 1

train_ds = LanguageModelDataset('./data/masnavi.txt', seq_len, min_count)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

x, y = train_ds[0]

for in_token, out_token in zip(x, y):
    print('{:2d} -> {:2d}'.format(in_token, out_token))

# Hyper-parameters
embed_size = 128
hidden_size = 256
num_layers = 1

num_epochs = 20
num_samples = 200 # number of words to be sampled
learning_rate = 0.0002

#RNN For Language Modeling
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.out = nn.Linear(2*hidden_size, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.out.weight.data.uniform_(-0.1, 0.1)
        self.out.bias.data.fill_(0)
        
        
#     def init_hidden(self):
#         h = torch.zeros(())
        
    def forward(self, input, hidden):
        # embed word ids to vectors
        output = self.embedding(input)
        print(output.size())
        
        # forward LSTM step
        output, hidden = self.lstm(output, hidden)
        print(output.size())
        
        # reshape output to (bs * seq_length, hidden_size)
        output = output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        print(output.size())
        
        # decode hidden states of all time steps
        output = self.out(output)
        print(output.size())
        
        return output, hidden

vocab_size = len(train_ds.vocabulary)
model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
if use_gpu:
    criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # initial hidden and state memory (h, c)
    states = (to_var(torch.zeros(2*num_layers, batch_size, hidden_size)),  # * 2 -> because of bidirectional
              to_var(torch.zeros(2*num_layers, batch_size, hidden_size)))
    
#     for i in range(0, ids.size(1) - seq_length, seq_length):
#         # get a batch
#         inputs = to_var(ids[:, i: i + seq_length])
#         targets = to_var(ids[:, (i + 1): (i + 1) + seq_length].contiguous())
        
    for step, (inputs, targets) in enumerate(train_dl):
        inputs = to_var(inputs)
        
        # Forward
        states = detach(states)
        outputs, states = model(inputs, states)
        _, preds = torch.max(outputs.data, dim=1)
        print(preds.size())
#         pdb.set_trace()
        
        # loss
        print(targets.view(-1))
        loss = criterion(preds, targets.view(-1))
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        
        # report
        if i % 100 == 0:
            sys.stdout.flush()
            sys.stdout.write('\rEpoch [%2d/%2d] | Step [%3d/%3d] | Loss: %.3f, Perplexity: %5.2f' % 
                  (epoch + 1, num_epochs, step, num_batches, loss.data[0], np.exp(loss.data[0])))

torch.save(model.state_dict(), f'masnavi-bi-{num_layers}-layers-{embed_size}-{hidden_size}-perplexity-{np.exp(loss.data[0])}.pth')

with open(sample_path, 'w', encoding='utf8') as f:
    state = (to_var(torch.zeros(2*num_layers, 1, hidden_size)),
             to_var(torch.zeros(2*num_layers, 1, hidden_size)))
    
    # select a random word id to start sampling
    prob = torch.ones(vocab_size)
    input = to_var(torch.multinomial(prob, num_samples=1).unsqueeze(1), volatile=True)
    
    
    for i in range(num_samples):
        output, state = model(input, state)
        
        # Sample an id
        prob = output.squeeze().data.exp().cpu()
        word_id = torch.multinomial(prob, 1)[0]
        
        # Feed sampled word id to next time step
        input.data.fill_(word_id)
        
        # write to file
        word = corpus.vocabulary.index2word[word_id]
        word = '\n' if word == '<EOS>' else word + ' '
        f.write(word)
        
        if (i + 1) % 100 == 0:
            print('Sampled [%3d/%3d] words and saved to %s' % (i + 1, num_samples, sample_path))

print(open(sample_path, encoding='utf8').read())
