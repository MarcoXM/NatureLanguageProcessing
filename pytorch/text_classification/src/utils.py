import torch
from torch.utils.data import DataLoader

def train_fn(dataLoader,model,optimizer,scheduler,criterion,device):

    # Train the model
    train_loss = 0
    train_acc = 0
    model.train()
    for i, (text, offsets, cls) in enumerate(dataLoader):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        b = cls.size(0)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(dataLoader), train_acc / len(dataLoader)/b

def evaluate_fn(dataLoader,model,criterion,device):

    # Train the model
    valid_loss = 0
    valid_acc = 0
    model.eval()
    for i, (text, offsets, cls) in enumerate(dataLoader):
        
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        b = cls.size(0)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == cls).sum().item()


    return valid_loss / len(dataLoader), valid_acc / len(dataLoader)/b


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label