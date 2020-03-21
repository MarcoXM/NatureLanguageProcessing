import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os

def get_dataset():
    if not os.path.isdir('./.data'):
        os.mkdir('./.data')
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
        root='./.data', ngrams=NGRAMS, vocab=None)
    return train_dataset,test_dataset


if __name__ == '__main__':
    train_dataset,test_dataset = get_dataset()
    print("dataset get !!!")    
