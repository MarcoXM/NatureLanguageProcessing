import torch
from model import RNN
import argparse
import string


def categoryTensor(idx):
        tensor = torch.zeros(1, n_categories)
        tensor[0][idx] = 1
        return tensor
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

max_length = 20
WEIGHT_PATH = "./weights/text_gen11.23178537686666.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1
categories = ['Arabic', 'Chinese', 'Korean', 'Japanese', 'French', 'English', 'Czech', 'Irish', 'Portuguese', 'German', 'Scottish', 'Polish', 'Italian', 'Vietnamese', 'Dutch', 'Spanish', 'Russian', 'Greek']
n_categories = len(categories)
cate2index = {v:i for i,v in enumerate(categories)}
rnn = RNN(categories,n_letters, 128, n_letters)
checkpoint = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))
rnn.load_state_dict(checkpoint)
rnn.to(device)

# Sample from a category and starting letter
def sample(category, start_letter='A',rnn = rnn):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(cate2index[category])
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()
        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
        return output_name

# Get multiple samples from one category and multiple starting letters
def multi_samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))
    print("All names has generated !!")


parser = argparse.ArgumentParser(
    description='Text_genneration With Pytorch')
parser.add_argument("--lang_cate", default="Spanish", type=str, help="inference text")
args = parser.parse_args()
print(multi_samples(category=args.lang_cate, start_letters='CHIV'))
