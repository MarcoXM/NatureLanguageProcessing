import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import nltk
import numpy as np
import random
from collections import Counter
import pandas as pd
flatten = lambda l: [item for sublist in l for item in sublist]

class Skigram(nn.Module):
    def __init__(self,vocab_size,vec_dim):
        super(Skigram,self).__init__()

        self.embeding_v = nn.Embedding(vocab_size,vec_dim)
        self.embeding_u = nn.Embedding(vocab_size,vec_dim) # Two matrice here and it is what we want.

        nn.init.xavier_uniform_(self.embeding_u.weight,gain=1)
        nn.init.xavier_uniform_(self.embeding_u.weight,gain=1) # Initialize it

        # And you could treat this embeding layer as a lookup table.

    def forward(self,center_word,target_word,outsiede_words):

        emb_center = self.embeding_v(center_word) # this is the center word embeding
        emb_target = self.embeding_u(target_word)
        emb_outside = self.embeding_u(outsiede_words) # Here are several words

    ### This time we return loss directly !

        numerator = emb_target.bmm(emb_center.transpose(1,2)).squeeze(2) # the dot product of the same word' embedding 
        # should be high!!!
        denominator = emb_outside.bmm(emb_center.transpose(1,2)).squeeze(2)

        loss = - torch.mean(torch.log(torch.exp(numerator)/torch.sum(torch.exp(denominator), dim=1).unsqueeze(1)))

        return loss

    
    def predict(self,word):
        return self.embeding_v(word)



if __name__ == '__main__':
    sk = Skigram(1000,300)
    print(sk)