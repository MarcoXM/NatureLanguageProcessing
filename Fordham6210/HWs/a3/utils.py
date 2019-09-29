

from bs4 import BeautifulSoup
from urllib.request import urlopen
import numpy as np
import warnings
import os 
import re
import pandas as pd


def getCorpus(link,tag):
    html = urlopen(link).read().decode('utf-8','ignore')
    soup = BeautifulSoup(html, features='lxml')
    all_href = soup.find_all(tag)
    
    corpus = []
    for i in all_href:
        text =  i.get_text()
        cleaned = re.sub(r'\n','',text)
        corpus.append(cleaned)
    
    return corpus

if __name__=='__main__':
    positivecorpus = getCorpus('https://storm.cis.fordham.edu/~yli/data/electronics/positive.review','review_text')
    print('done')