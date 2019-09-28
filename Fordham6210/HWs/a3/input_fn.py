

from collections import Counter

class PoetDataset(object):
    def __init__(self,filepath):
        super(PoetDataset).__init__()
        self.data = self._read_data(filepath)
        self.authorCnt = Counter(self.data['Author'].values)

    def _read_data(self.datapath):
        return pd.read_csv(datapath)

    def top_k(self,k):
        top_authors = self.authorCnt.most_common(k)

        # return top k list 数量一样就字母表排序
        pass
    