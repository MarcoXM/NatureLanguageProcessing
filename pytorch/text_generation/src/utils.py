import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(model,dataloader,criterion,optimizer,device):
    total_loss = 0
    for data in dataloader:
        category_tensor, input_line_tensor, target_line_tensor = data
        # print(category_tensor.size(), input_line_tensor.size(), target_line_tensor.size())
        category_tensor.to(device)
        input_line_tensor.to(device)
        target_line_tensor.to(device)
        target_line_tensor.squeeze_(0)
        target_line_tensor.unsqueeze_(-1)
        hidden = model.initHidden()
        model.to(device)
        model.train()
        model.zero_grad()
        loss = 0
        for i in range(input_line_tensor.size(1)):
            # print(category_tensor.squeeze(0).size(), input_line_tensor[0][i].size(), hidden.size())
            output, hidden = model(category_tensor.squeeze(0), input_line_tensor[0][i], hidden)
            # print(output, target_line_tensor[i])
            # print(output.size(), target_line_tensor[i].size())
            l = criterion(output, target_line_tensor[i])
            loss += l
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return output, total_loss / len(dataloader)
