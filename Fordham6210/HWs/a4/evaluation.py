from utils import prepare_word
from utils import prepare_seq
import torch.nn.functional as F


def most_similiar(model,word,vocab,word2idx,topk):
    vec = model.predict(prepare_word(word,word2idx))

    similarity = []

    for i in range(len(vocab)):
        if vocab[i] == word:
            continue

        vec_test = model.predict(prepare_word(list(vocab)[i],word2idx))

        cos_distance = F.cosine_similarity(vec,vec_test).item()[0]
        similarity.append(cos_distance)

    return sorted(similarity,key=lambda x:x[1],reverse=True)[:topk]