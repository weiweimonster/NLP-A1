from torch.utils.data import Dataset
from util import *
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List, Dict

classes = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
           "comp.windows.x", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey",
           "sci.crypt", "sci.electronics", "sci.med", "sci.space", "misc.forsale", "talk.politics.misc",
           "talk.politics.guns", "talk.politics.mideast", "talk.religion.misc", "alt.atheism",
           "soc.religion.christian"]


class newsgroup_dataset(Dataset):
    def __init__(self, data, word_to_idx, max_len=None, ngram=1):
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        self.n = ngram
        self.class_label = {w: i for i, w in enumerate(classes)}
        self.data, self.labels = self.convert_to_idx(data)

    def convert_to_idx(self, data: List[Tuple[Any, Any]]):
        wix = self.word_to_idx
        labels = []
        data_points = []
        for d, label in data:
            temp = []
            for i in range(1, self.n + 1):
                line = generate_ngrams(d, i)
                for w in line:
                    if w in wix:
                        temp.append(wix[w])
            if self.max_len is not None and len(temp) != self.max_len:
                if len(temp) > self.max_len:
                    temp = temp[:self.max_len]
                else:
                    temp = temp + [0] * (self.max_len - len(temp))
                data_points.append(torch.tensor(temp))
            else:
                data_points.append(torch.tensor(temp))
            labels.append(self.get_class_id(label))
        padded_data = pad_sequence(data_points, batch_first=True, padding_value=0)

        return padded_data, torch.tensor(labels)

    def get_class_id(self, class_name: str) -> int:
        return self.class_label[class_name]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
