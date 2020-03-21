import torch
import torch.nn as nn
import string


class RNN(nn.Module):
    def __init__(self, categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.categories = categories
        self.n_categories = len(self.categories)
        self.i2h = nn.Linear(self.n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(self.n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.softmax = nn.LogSoftmax(dim=1) # logsoft max have utilize while we have a great amount 

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                output_ = dropout(output)
            else:
                output_ += dropout(output)
        output_ /= len(self.dropouts)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

if __name__ == "__main__":

    categories = ['Arabic', 'Chinese', 'Korean', 'Japanese', 'French', 'English', 'Czech', 'Irish', 'Portuguese', 'German', 'Scottish', 'Polish', 'Italian', 'Vietnamese', 'Dutch', 'Spanish', 'Russian', 'Greek']
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters) + 1
    rnn = RNN(categories, n_letters, 128, n_letters)

    print(rnn)