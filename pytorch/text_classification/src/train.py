import torch
from dataset import get_dataset
from model import TextSentiment
from utils import generate_batch,train_fn,evaluate_fn
import time
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():

    device = "gpu" if torch.cuda.is_available() else "cpu"
    train_dataset,test_dataset = get_dataset()
    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = 32
    NUN_CLASS = len(train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
    BATCH_SIZE = 16
    N_EPOCHS = 5
    min_valid_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss().to(device) # mutil-class use the CrossEntropy
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    train_loader = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    valid_loader = DataLoader(sub_valid_, batch_size=BATCH_SIZE,
                      collate_fn=generate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                      collate_fn=generate_batch)

    for epoch in tqdm(range(N_EPOCHS)):

        start_time = time.time()
        train_loss, train_acc = train_fn(dataLoader=train_loader, model=model,optimizer=optimizer,scheduler=scheduler,criterion=criterion,device=device)
        valid_loss, valid_acc = evaluate_fn(dataLoader=valid_loader,model=model,criterion=criterion,device=device)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        if valid_loss < min_valid_loss:
            torch.save(model.state_dict(),"../weights/text_news{}.pth".format(valid_loss))
            print(min_valid_loss ,"--------->>>>>>>>",valid_loss)
            min_valid_loss = valid_loss

    print('Checking the results of test dataset...')
    test_loss, test_acc = evaluate_fn(dataLoader=test_loader,model=model,criterion=criterion,device=device)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

if __name__ == "__main__":
    main()
      