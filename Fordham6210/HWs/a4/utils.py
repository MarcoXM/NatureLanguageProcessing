import re
import nltk
import numpy as np

def prepare_word(word,word2idx):

    '''
    getting the int of the word! 
    '''
    return np.array(word2idx[word] if word2idx.get(word) is not None else word2idx['<UNK>'])



def prepare_seq(seq,word2idx):

    '''Transform a seq into ints '''

    return list(map(lambda w: word2idx[w] if word2idx.get(w) is not None else word2idx['<UNK>'],seq))


def get_most_author(authors,topk):
    from collections import Counter
    cnt = Counter(authors)

    return cnt.most_common(topk)

def clean(text):
    l = re.sub(r'[\[L\]|\[P\]]', '', text)
    return re.sub(r'[^\w]', ' ', l)

def get_corpus(df,name):

    a = df[df['Author'] == name][['Body']]
    a['Body1'] =a['Body'].apply(lambda x : clean(x)) 
    corpus = []
    b = a.Body1.tolist()
    for poet in b:
        tokens = nltk.word_tokenize(poet)
        corpus.append(tokens)

    return corpus