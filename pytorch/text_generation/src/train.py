from dataset import TextDataset,get_data,n_letters
from utils import train,timeSince
from model import RNN
import torch
import torch.nn as nn
import time
from tqdm import tqdm


n_iters = 10000
print_every = 1000
train_loss = [] # Reset every plot_every iters

start = time.time()
device = "gpu" if torch.cuda.is_available() else "cpu"
data = get_data(data_path = 'data/names/*.txt')
rnn = RNN(data['all_categories'],n_letters, 128, n_letters)
dataset = TextDataset(data)
train_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
criterion = nn.NLLLoss()
optimizer = torch.optim.AdamW(rnn.parameters(), lr=0.0005)

for iter in tqdm(range(1, n_iters + 1)):
    output, loss = train(rnn,train_loader,criterion,optimizer,device)
    train_loss.append(loss)

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter% 2000 == 0:
        torch.save(rnn.state_dict(),"./weights/text_gen{}.pth".format(loss))
