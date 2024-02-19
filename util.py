import string
from typing import List, Any, Tuple, Dict

from newsgroups import newsgroups_data_loader
from sst2 import sst2_data_loader
import csv
import re
from collections import Counter, defaultdict
import random


def process_data(train, val, dev, test, ngram=1):
    vocab = build_vocab(train, val, ngram)
    word_to_idx = create_word_to_idx(vocab)

    encoded_train = encode_data(train, word_to_idx, ngram)
    encoded_val = encode_data(val, word_to_idx, ngram)

    add_bias(encoded_train, len(vocab) + 1)
    add_bias(encoded_val, len(vocab) + 1)

    encoded_dev = encode_data(dev, word_to_idx, ngram)
    add_bias(encoded_dev, len(vocab) + 1)

    encoded_test = encode_data(test, word_to_idx, ngram)
    add_bias(encoded_test, len(vocab) + 1)

    return encoded_train, encoded_val, encoded_dev, encoded_test, word_to_idx


def encode_data(data, word_to_idx, ngram=1):
    ret = []
    for d, label in data:
        counter = Counter(generate_ngrams(d, ngram))
        encoded = defaultdict(int)
        for k, v in counter.items():
            if k in word_to_idx:
                encoded[word_to_idx[k]] = v
            else:
                encoded[0] += v
        ret.append([encoded, label])
    return ret


def save_results(predictions: List[Any], results_path: str) -> None:
    """ Saves the predictions to a file.

    Inputs:
        predictions (list of predictions, e.g., string)
        results_path (str): Filename to save predictions to
    """
    # TODO: Implement saving of the results.
    header = ['id', 'label']
    n = len(predictions)
    ids = [str(i) for i in range(n)]
    rows = zip(ids, predictions)
    with open(results_path, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def compute_accuracy(labels: List[Any], predictions: List[Any]) -> float:
    """ Computes the accuracy given some predictions and labels.

    Inputs:
        labels (list): Labels for the examples.
        predictions (list): The predictions.
    Returns:
        float representing the % of predictions that were true.
    """
    if len(labels) != len(predictions):
        raise ValueError("Length of labels (" + str(len(labels)) + " not the same as " \
                                                                   "length of predictions (" + str(
            len(predictions)) + ".")
    # TODO: Implement accuracy computation.
    correct = 0
    for i in range(len(labels)):
        if int(labels[i]) == predictions[i]:
            correct += 1
    return correct / len(labels)


def build_vocab(train, val, ngram=1):
    word_list = []
    for data, label in train:
        word_list += generate_ngrams(data, ngram)
    # for data, label in val:
    #     word_list += generate_ngrams(data, ngram)
    return Counter(word_list)


def create_word_to_idx(word_dict):
    temp = [(w, i + 1) for i, w in enumerate(word_dict)]
    counter = Counter()
    dict.update(counter, temp)
    return counter


def generate_ngrams(data, n: int = 1):
    file = open("stopwords.txt", 'r')
    stop_words = set([line for line in file])
    file.close()
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
    rare_words = {"maxaxaxaxaxaxaxaxaxaxaxaxaxaxax", "db", "pts", "pt", "oo", "aaa", "aa", "[", "]", "\t"}
    pattern = "[" + string.punctuation + "\t" + "]"
    data =  re.sub(pattern, '', data)
    stop_words.update()
    tokens = []
    for line in data.splitlines():
        for token in line.strip().split(" "):
            if token != "" and token not in stop_words and token not in rare_words:
                tokens.append(token)
    # tokens = [token.lower() for token in data.strip().split(" ") if token != "" and token not in stop_words]
    # tokens = [token for token in data.split(" ") if token != ""]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]



# generate the bag of words from data
# if mode is false then means no validation data
# return each data points' bag of word in a list
def generate_bow(data, val=None, n=1, mode=False):
    list_of_dict = []
    for d, label in data:
        counter = Counter(generate_ngrams(d, n))
        list_of_dict.append([counter, label])
    if not mode:
        return list_of_dict
    for d, label in val:
        counter = Counter(generate_ngrams(d, n))
        list_of_dict.append([counter, label])
    random.shuffle(list_of_dict)
    return list_of_dict


def get_num_classes(data_type: str) -> int:
    if data_type == "newsgroups":
        return 20
    elif data_type == "sst2":
        return 2


def get_num_feature(list_of_dict: List[List[Counter[str] | Any]]) -> tuple[
    list[list[dict[int, Any] | Any]], dict[Any, int]]:
    w = {}
    count = 0
    ret = []
    for dict, label in list_of_dict:
        new_dict = {}
        for k in dict.keys():
            if k not in w:
                w[k] = count
                count += 1
            new_dict[w[k]] = dict[k]
        ret.append([new_dict, label])
    return ret, w


def add_bias(data, num_features):

    for dict, label in data:
        dict[num_features] = 1


def evaluate(model: Any, data: List[Tuple[Any, Any]], results_path: str, loader=False, convert=False) -> float:
    """ Evaluates a dataset given the model.

    Inputs:
        model: A model with a prediction function.
        data: Suggested type is (list of pair), where each item is a training
            examples represented as an (input, label) pair. And when using the
            test data, your label can be some null value.
        results_path (str): A filename where you will save the predictions.
    """

    if model.get_mode() == 1:
        labels = [model.convert_class(d[1]) for d in data]
    elif model.get_mode() == 0:
        labels = [example[1] for example in data]
    else:
        labels = [example[1].tolist() for example in data]
    if loader:
        labels = []
        predictions = []
        for d, l in data:
            pred = model.forward(d).argmax(dim=-1)
            predictions += pred.tolist()
            labels += l.tolist()
    else:
        predictions = [model.predict(example[0]) for example in data]
    if convert:
        predictions = model.id_to_class(predictions)

    save_results(predictions, results_path)

    # return compute_accuracy(labels, predictions)


def load_data(data_type: str, feature_type: str, model_type: str):
    """ Loads the data.

    Inputs:
        data_type: The type of data to load.
        feature_type: The type of features to use.
        model_type: The type of model to use.
        
    Returns:
        Training, validation, development, and testing data, as well as which kind of data
            was used.
    """
    data_loader = None
    if data_type == "newsgroups":
        data_loader = newsgroups_data_loader
    elif data_type == "sst2":
        data_loader = sst2_data_loader

    assert data_loader, "Choose between newsgroups or sst2 data. " \
                        + "data_type was: " + str(data_type)

    # Load the data. 
    train_data, val_data, dev_data, test_data = data_loader("data/" + data_type + "/train/train_data.csv",
                                                            "data/" + data_type + "/train/train_labels.csv",
                                                            "data/" + data_type + "/dev/dev_data.csv",
                                                            "data/" + data_type + "/dev/dev_labels.csv",
                                                            "data/" + data_type + "/test/test_data.csv",
                                                            feature_type,
                                                            model_type)

    return train_data, val_data, dev_data, test_data


def convert_feature(data, feature_dict):
    ret = []
    count = 0
    for d, label in data:
        new_dict = {}
        for k in d.keys():
            if k in feature_dict:
                new_dict[feature_dict[k]] = d[k]
            else:
                count += 1
        ret.append([new_dict, label])
    # print(count)
    return ret


def top_k_freq(vocab, top_k):
    counter = Counter()
    count = 0
    for k, v in vocab.most_common():
        if count == top_k:
            return counter
        counter[k] = v
        count += 1
    return counter


def top_k_words(vocab, min_f):
    counter = Counter()
    for k, v in vocab.most_common():
        if v >= min_f:
            counter[k] = v
    return counter
