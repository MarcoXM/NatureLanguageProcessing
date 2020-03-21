from model import TextSentiment
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from dataset import get_dataset
import pickle
import argparse
import torch

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}
WEIGHT_PATH = "../weights/text_news0.2672930294473966.pth"


vocab = pickle.load(open(".data/save_vocab.p", "rb"))

device = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 1308844
EMBED_DIM = 32
NUM_CLASS = 4
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS)
checkpoint = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.to(device)
def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


parser = argparse.ArgumentParser(
    description='Text_classification With Pytorch')
parser.add_argument("--text", default="MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind.", type=str, help="inference text")
args = parser.parse_args()
print("This is a %s news" %ag_news_label[predict(args.text, model, vocab, 2)])