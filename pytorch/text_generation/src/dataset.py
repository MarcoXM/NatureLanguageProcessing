from io import open
import glob
import os
import unicodedata
import string
import random
import torch

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): 
    return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category

def get_data(data_path):
    category_lines = {}
    all_categories = []
    for filename in findFiles(data_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')

    print('# categories:', n_categories, all_categories)
    print(unicodeToAscii("O'Néàl"))
    print(all_categories)
    return {
        'all_categories':all_categories,
        'category_lines':category_lines
    }
# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair(all_categories,category_lines):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


class TextDataset:
    def __init__(self, dataset):
        self.all_categories = dataset['all_categories']
        self.category_lines = dataset['category_lines']
        self.n_categories = len(self.all_categories)
        self.all_letters = string.ascii_letters + " .,;'-"
        self.n_letters = len(all_letters) + 1

    def __len__(self):
        return len(self.all_categories)
    
    def __getitem__(self,idx):
        category = self.all_categories[idx]
        line = randomChoice(self.category_lines[category])

        category_tensor = self.categoryTensor(idx)
        input_line_tensor = self.inputTensor(line)
        target_line_tensor = self.targetTensor(line)

        return category_tensor, input_line_tensor, target_line_tensor

    def categoryTensor(self,idx):
        tensor = torch.zeros(1, self.n_categories)
        tensor[0][idx] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(self,line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.all_letters.find(letter)] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(self,line):
        letter_indexes = [self.all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(self.n_letters - 1) # EOS
        return torch.LongTensor(letter_indexes)





if __name__ == '__main__':
    dataset = get_data('data/names/*.txt')
    print("dataset get !!!")
    print(randomTrainingPair(dataset['all_categories'],dataset['category_lines'])) # It would return one language type and related name!
    train_dataset = TextDataset(dataset)
    print(iter(train_dataset))