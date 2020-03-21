import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim 
import torch.nn.functional as F
import nltk
import numpy as np
import random
from collections import Counter
import pandas as pd
from torch.utils.data import DataLoader
from utils import prepare_word
flatten = lambda l: [item for sublist in l for item in sublist]

class Dataset(nn.Module):
    def __init__(self,window_size):
        super(Dataset,self).__init__()
        self.ws = window_size
        self.corpus = None
        self.word2idx = None
        self.idx2word = None
        self.ds = None



    def update(self,corpus):
        self.corpus = [[word.lower() for word in sent]for sent in corpus]

        ## vocabulary 
        vocab = list(set(flatten(corpus)))
        vocab.append('<UNK>') # Yes we have unknow word!
        self.vocab = vocab
        word2idx = {'<UNK>' : 0}

        for w in vocab:
            if word2idx.get(w) is None:
                word2idx[w] = len(word2idx)

        self.word2idx= word2idx
        self.idx2word = {v:k for k,v in word2idx.items()}

        windows_word_pair = flatten(list(nltk.ngrams(['<DUMMY>'] * self.ws + center_word + ['<DUMMY>'] * self.ws, self.ws *2 +1 ) for center_word in self.corpus))

        train_x,train_y = [],[]

        for data in windows_word_pair:
            for w in range(self.ws * 2 + 1):
                if w== data[self.ws] or data[w] == '<DUMMY>':
                    continue
                train_x.append(prepare_word(data[self.ws],self.word2idx))
                train_y.append(prepare_word(data[w],self.word2idx))
        ds = [(x,y) for x,y in zip(train_x,train_y)]
        self.ds = ds
        return self.ds

    

def poetloader(dataset,batchsize):
    return DataLoader(dataset,batch_size = batchsize)


    
