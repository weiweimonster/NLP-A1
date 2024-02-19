from torch.utils.data import Dataset
from util import *
import torch
from torch.nn.utils.rnn import pad_sequence


class sentiment_dataset(Dataset):
    def __init__(self, data, word_to_idx, max_len=None, ngram=1):
        self.word_to_idx = word_to_idx
        self.n = ngram
        self.max_len = max_len
        self.data, self.labels = self.convert_to_idx(data)

    def convert_to_idx(self, data):
        word_to_idx = self.word_to_idx
        labels = []
        data_points = []

        for d, label in data:
            temp = []
            for i in range(1, self.n + 1):
                line = generate_ngrams(d, i)
                for w in line:
                    if w in word_to_idx:
                        temp.append(word_to_idx[w])
            if self.max_len is not None and len(temp) != self.max_len:
                if len(temp) > self.max_len:
                    temp = temp[:self.max_len]
                else:
                    temp = temp + [0] * (self.max_len - len(temp))
                data_points.append(torch.tensor(temp))
            else:
                data_points.append(torch.tensor(temp))
            labels.append(int(label))
        padded_data = pad_sequence(data_points, batch_first=True, padding_value=0)

        return padded_data, torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
