
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import sys
sys.path.append('../')
import torch

from build_lang import Lang

EOS_token = 0
SOS_token = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_lang(path,lang1,lang2,reverse =False):
    print('Starting ! ! !')
    lines = open(path +'%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
     # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10 # 10 words at large !

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH and pair[0].startswith(eng_prefixes)
    #getting those pairs len(Eng) <10 and start with...
    
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_lang('/Users/marcowang/NatureLanguageProcessing/pytorch/data/',lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)

    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.num_words)
    print(output_lang.name, output_lang.num_words)
    return input_lang, output_lang, pairs

def sentence2idx(lang,sentence):
    return [lang.word2idx[word] for word in sentence.split(' ')]

def sentenc2tensor(lang,sentence):
    idx = sentence2idx(lang,sentence)
    idx.append(EOS_token) ##  Adding the ending word!
    return torch.LongTensor(idx).view(-1, 1).to(DEVICE)

def pair2tensor(lang1,lang2,pair):
    
    x_tensor = sentenc2tensor(lang1,pair[0]) # input language >>>>>> index
    y_tensor = sentenc2tensor(lang2,pair[1]) # output language >>>>>>> index
    
    return x_tensor,y_tensor




if __name__ == '__main__':
    eng_lang,fra_lang, pair = read_lang('/Users/marcowang/NatureLanguageProcessing/pytorch/data/','eng','fra')
    input_lang, output_lang, pairs = prepareData(lang1='eng', lang2='fra')

    print('test done !')
