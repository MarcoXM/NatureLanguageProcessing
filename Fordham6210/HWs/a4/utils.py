

def prepare_word(word,word2idx):

    '''
    getting the int of the word! 
    '''
    return word2idx[word] if word2idx.get(word) is not None else word2idx['<UNK>']



def prepare_seq(seq,word2idx):

    '''Transform a seq into ints '''

    return list(map(lambda w: word2idx[w] if word2idx.get(w) is not None else word2idx['<UNK>'],seq))