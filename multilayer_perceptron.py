""" Multi-layer perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function headers.
"""
import os
import sys
import argparse

from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from util import *
from SSTDataset import sentiment_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def get_accuracy(prediction, label):
    b, _ = prediction.shape
    pred_class = prediction.argmax(dim=-1)
    correct = pred_class.eq(label).sum()
    return correct / b


class MultilayerPerceptronModel(nn.Module):
    """ Multi-layer perceptron model for classification.
    """

    def __init__(self, output_dim, v_size, embedding_dim=10, h1=256, h2=128, h3=64, h4=32, drop_out=0.4,
                 max_length=1):
        """ Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        # TODO: Implement initialization of this model.
        super(MultilayerPerceptronModel, self).__init__()
        # Note: You can add new arguments, with a default value specified.
        self.mode = 3
        self.max_length = max_length
        self.emb = nn.Embedding(v_size, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.dropout1 = nn.Dropout()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, h1),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(h2, output_dim)
        )
        # self.fc1 = nn.Linear(embedding_dim * max_length, h1)
        # self.fc2 = nn.Linear(h1, h2)
        # self.fc3 = nn.Linear(h2, h3)
        # self.fc4 = nn.Linear(h3, h4)
        # self.fc5 = nn.Linear(h4, output_dim)

    def forward(self, text: Tensor):
        emb_x = self.emb(text)
        x = emb_x.view(emb_x.shape[0], -1)
        x = self.mlp(x)
        # x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc3(x))
        # x = self.dropout1(x)
        # x = self.relu(self.fc4(x))
        # x = self.dropout1(x)
        # x = self.fc5(x)
        return x

    def get_mode(self):
        return self.mode

    def predict(self, model_input: torch.Tensor):
        """ Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.    

        """
        # TODO: Implement prediction for an input.
        self.eval()
        with torch.no_grad():
            if model_input.shape[1] < self.max_length:
                zeros = torch.zeros(model_input.shape[0], self.max_length - model_input.shape[1])
                model_input = torch.cat([model_input, zeros], dim=0)
            model_input = model_input[:, :self.max_length]
            out_put = self.forward(model_input)
        return out_put.argmax(dim=-1)

    def learn(self, training_data, validation_data, loss_fct, optimizer, num_epochs, lr) -> None:
        """ Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is 
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            loss_fct: The loss function.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        # TODO: Implement the training of this model.
        epoch_losses = []
        epoch_accuracy = []
        for epoch in range(num_epochs):
            self.train()
            losses = []
            accs = []
            val_losses = []
            val_accs = []
            for data, label in tqdm(training_data, desc="training..."):
                output = self.forward(data)
                loss = loss_fct(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                accuracy = get_accuracy(output, label)
                accs.append(accuracy.item())
            self.eval()
            with torch.no_grad():
                for v_data, v_label in tqdm(validation_data, desc="evaluating..."):
                    prediction = self.forward(v_data)
                    val_loss = loss_fct(prediction, v_label)
                    val_accuracy = get_accuracy(prediction, v_label)
                    val_losses.append(val_loss.item())
                    val_accs.append(val_accuracy.item())
            print(f"epoch: {epoch}")
            print(f"train_loss: {np.mean(losses):.3f}, train_acc: {np.mean(accs):.3f}")
            print(f"valid_loss: {np.mean(val_losses):.3f}, valid_acc: {np.mean(val_accs):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiLayerPerceptron model')
    parser.add_argument('-d', '--data', type=str, default='sst2',
                        help='Dataset')
    parser.add_argument('-f', '--features', type=str, default='feature_name', help='Feature type')
    parser.add_argument('-m', '--model', type=str, default='mlp', help='Model type')
    args = parser.parse_args()

    data_type = args.data
    feature_type = args.features
    model_type = args.model
    ngram = 3
    train_params = {'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 6}

    other_params = {'batch_size': 16,
                    'shuffle': False,
                    'num_workers': 6}
    num_epochs = 3
    lr = 0.01
    max_len = 35
    min_freq = 2
    train_data, val_data, dev_data, test_data = load_data(data_type, feature_type, model_type)
    vocab = build_vocab(train_data, val_data, 1)
    # vocab_2gram = build_vocab(train_data, val_data, 2)
    # vocab_3gram = build_vocab(train_data, val_data, 3)
    total_vocab = vocab
    total_vocab = top_k_freq(total_vocab, 1000)
    word_to_idx = create_word_to_idx(total_vocab)
    # 35, 800 0.72

    train_dataset = sentiment_dataset(train_data, word_to_idx, max_len=max_len, ngram=ngram)
    train_dataloader = DataLoader(train_dataset, **train_params)

    val_dataset = sentiment_dataset(val_data, word_to_idx, max_len=max_len, ngram=ngram)
    val_dataloader = DataLoader(val_dataset, **other_params)

    dev_dataset = sentiment_dataset(dev_data, word_to_idx, max_len=max_len, ngram=ngram)
    dev_dataloader = DataLoader(dev_dataset, **other_params)

    test_dataset = sentiment_dataset(test_data, word_to_idx)
    test_dataloader = DataLoader(test_dataset, **other_params)

    vocab_size = len(total_vocab)
    num_classes = get_num_classes(data_type)

    # Train the model using the training data.
    model = MultilayerPerceptronModel(num_classes, vocab_size + 1, 25, 128, 64, 64, 32, 0.4, max_len)

    print("Training the model...")
    # Note: ensure you have all the inputs to the arguments.
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.learn(train_dataloader, val_dataloader, loss_function, optimizer, num_epochs, lr)

    # Predict on the development set. 
    # Note: if you used a dataloader for the dev set, you need to adapt the code accordingly.
    dev_accuracy = evaluate(model,
                            dev_dataloader,
                            os.path.join("results", f"mlp_{data_type}_{feature_type}_dev_predictions.csv"), loader=True)

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    # evaluate(model,
    #          test_data,
    #          os.path.join("results", f"mlp_{data_type}_{feature_type}_test_predictions.csv"))
