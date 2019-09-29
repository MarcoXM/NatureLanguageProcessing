SPECIAL_WORDS = {'PADDING': '<PAD>'}
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer 
from collections import Counter
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer 
import re
import pandas as pd
flatten = lambda l: [item for sublist in l for item in sublist]

class Dataset(object):
    def __init__(self,corpus,tokenizer,stemmer,
                 remove_stop = True, 
                 keep_pantuation = True):
        super(Dataset,self)
        self.token_lookup = self._lookup()
        self.vocab2idx = None
        self.idx2vocab = None
        self.word_counter = None
        self._flatten = None
        self.tokenizer = tokenizer
        self.tokens = None
        self.int_text = None
        self.stemmer = stemmer
        self.keep_pantuation = keep_pantuation
        self.remove_stop = remove_stop
        self.X = None
        self.corpus = self._update(corpus)
        
        
        
    def _lookup(self):
        ''' lookup table to keep the puntuation !'''
        answer = {'.' : '||period||',
                  ',' : '||comma||',
                  '"' : '||quotation_mark||',
                  ';' : '||semicolon||',
                  '!' : '||exclamation_mark||',
                  '?' : '||question_mark||',
                  '(' : '||left_Parentheses||',
                  ')' : '||right_Parentheses||',
                  #'\n': '||return||',
                  '-' : '||dash||'}
        return answer
        
    def _update(self,text):
        #text = flatten(text)
        #print(text)
        if self.keep_pantuation:
            text = [self.preprocessing(t) for t in text]
        else:
            text = [re.sub(r"[{}]+".format(string.punctuation),'',t)  for t in text]
        #print(text)
            
        text = [self.stemmer.stem(t.lower()) for t in text]

        tokens = [self.tokenizer(t) for t in text]
        
        
        if self.remove_stop:
            tokens = tokens = [self.remove(t) for t in tokens]
        self.X = tokens
        tokens = flatten(tokens)
        #print(len(tokens))
        self.tokens = tokens
        #print(tokens)
        self.word_counter = Counter(self.tokens)
        self.vocab2idx, self.idx2vocab = self.create_lookup_tables(text + list(SPECIAL_WORDS.values()))
        self.int_text = [self.vocab2idx[word] for word in text]
        
        return text
        
    def create_lookup_tables(self,text):
        vocab_to_int = { v:i+2 for i,v in enumerate(set(text))}
        vocab_to_int['<START>'] = 0
        vocab_to_int['<end>'] = 1
        int_to_vocab = { v:k for k,v in vocab_to_int.items()}
        # return tuple
        return (vocab_to_int, int_to_vocab)
            
            
    def remove(self,text):
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in text if word not in stopwords.words('english')]
        
        return filtered_words
        
            
    def preprocessing(self,text):
        for key, token in self.token_lookup.items():
            text = text.replace(key, ' {} '.format(token))
        return text
        
        
    def getmostwords(self,k):
        
        return sorted(self.word_counter.most_common(k), key=lambda x: x[1])
    
    def getToken(self):
        return self.tokens
    
    def getTokenset(self):
        return set(self.tokens) 
        
    def getdata(self,X,y):
        df = pd.DataFrame()
        df['review'] = X
        df['sentiment'] = y
        return df
    
    
if __name__ == '__main__':
    ps = PorterStemmer() 
    p = Dataset(corpus=positivecorpus, tokenizer=word_tokenize, stemmer=ps,keep_pantuation= False)
    print(p.getmostwords(20))