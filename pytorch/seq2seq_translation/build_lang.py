EOS_token = 0
SOS_token = 1

class Lang:
    def __init__(self,name):
        
        self.name = name
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {0:'EOS',1:'SOS'}
        self.num_words = 2
        
    def add_sentence(self,sentence):
        for word in sentence.split(' '):
            self.add_word(word)
            
    def add_word(self,word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.word2count[word] = 1
            self.idx2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
            

if __name__ == "__main__":

    france = Lang('fra')
    print('test done !!!')