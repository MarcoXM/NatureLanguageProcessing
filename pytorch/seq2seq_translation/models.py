


import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncodeRNN(nn.Module):  # This is default Encoder !!!!
    
    def __init__(self,input_size,hidden_size):
        super(EncodeRNN,self).__init__()
        self.hidden=hidden_size
        
        self.emb = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        
    def forward(self,x,hidden):
        embed = self.emb(x).view(1,1,-1)
        output,hidden = self.gru(embed,hidden)
        return output,hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden, device=DEVICE)  # Yes the hidden state is 3d tensor !!

class DecodeRNN(nn.Module): # Default Decoder without Attention! 
    def __init__(self,hidden,output):
        super(DecodeRNN,self).__init__()
        self.hidden = hidden
        
        self.emb = nn.Embedding(output,hidden) # word2vec dimension == hidden !
        self.gru = nn.GRU(hidden,hidden) # word to word ,same size
        self.fc = nn.Linear(hidden,output) # pick word from vocab!
        
    def forward(self,x,hidden):
        
        embed = self.emb(x).view(1,1,-1)
        output,hidden = nn.gru(embed,hidden)
        output = F.log_softmax(output,dim=1)
        return output,hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden, device=DEVICE)


class AttnDecoder(nn.Module): # Multiplicative Attention
    def __init__(self,hidden_size,output_size,dropout_rate = 0.2):
        super(AttnDecoder,self).__init__()
        
        self.hidden = hidden_size # this is word vecter dimension
        self.output = output_size # this is vocab size
        self.drop = nn.Dropout(dropout_rate)
        
        
        self.emb = nn.Embedding(output_size,hidden_size)
        
        self.rnn = nn.GRU(hidden_size,hidden_size) # <EOS> inital and as the first word vector!
        self.h2newoutput = nn.Linear(hidden_size * 2, hidden_size)
        self.toword = nn.Linear(hidden_size,output_size)
        
    def forward(self,x,hidden,encoder_matrix):
        
        emb_vecter = self.emb(x)
        #print(emb_vecter.size())
        score = F.softmax(torch.matmul(emb_vecter,torch.t(encoder_matrix)),dim=-1)
        #print(score)
        new_context = torch.matmul(score,encoder_matrix)
        #print(new_context.size())
        output,hidden = self.rnn(emb_vecter,hidden)
        hidden = torch.cat([new_context,output],dim = -1)
        #print(new_input.size())
        output = F.tanh(self.h2newoutput(hidden))
        output = self.drop(output)
        final_output = F.log_softmax(self.toword(output),dim = 2)
        return final_output,hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden, device=DEVICE)
        



class AttnDecoderRNN2(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

if __name__ == '__main__':
    encoder1 = models.EncodeRNN(input_lang.num_words, hidden_size).to(DEVICE)
    attn_decoder1 = models.AttnDecoderRNN2(hidden_size, output_lang.num_words, dropout_rate=0.1).to(DEVICE)
    print(attn_decoder1)


