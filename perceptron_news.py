""" Perceptron model for Assignment 1: Starter code.

You can change this code while keeping the function headers.
"""
import os
import random
import sys
import argparse
from typing import Dict, List

from util import *


class PerceptronModel():
    """ Perceptron model for classification.
    """

    def __init__(self, num_features: int, num_classes: int):
        """ Initializes the model.

        Inputs:
            num_features (int): The number of features.
            num_classes (int): The number of classes.
        """
        # TODO: Implement initialization of this model.
        # class->index of feature-> weights
        # [1 0  1  0 | 1  0 1  0 |  1  0  1 0] feature
        # [1 1 −1 −2 | 1 −1 1 −2 | −2 −1 −1 1] weights
        self.mode = 0
        self.weights: Dict[int, Dict[int, float]] = {}
        self.class_dict = None
        ################################################
        self.features = num_features
        self.labels = num_classes

        def zero_weight():
            for i in range(num_classes):
                dict = {}
                for j in range(num_features):
                    dict[j] = 0
                self.weights[i] = dict

        zero_weight()

    def score(self, model_input: Dict, class_id: int):
        """ Compute the score of a class given the input.

        Inputs:
            model_input (features): Input data for an example
            class_id (int): Class id.

        Returns:
            The output score.
        """
        # TODO: Implement scoring function.
        w = self.weights[class_id]
        score = 0
        for k, v in model_input.items():
            score += model_input[k] * w[k]
        return score

    def predict(self, model_input: Dict) -> int:
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example

        Returns:
            The predicted class.
        """
        # TODO: Implement prediction for an input.
        scores = [self.score(model_input, i) for i in range(self.labels)]
        return scores.index(max(scores))

    def update_parameters(self, model_input: Dict, prediction: int, target: int, lr: float) -> None:
        """ Update the model weights of the model using the perceptron update rule.

        Inputs:
            model_input (features): Input data for an example
            prediction: The predicted label.
            target: The true label.
            lr: Learning rate.
        """
        # TODO: Implement the parameter updates.
        if target == prediction:
            return
        pred_w = self.weights[prediction]
        target_w = self.weights[target]
        for k in model_input.keys():
            pred_w[k] -= lr * model_input[k]
            target_w[k] += lr * model_input[k]

    def learn(self, training_data, val_data, num_epochs, lr) -> None:
        """ Perceptron model training.

        Inputs:
            training_data: Suggested type is (list of tuple), where each item can be
                a training example represented as an (input, label) pair or (input, id, label) tuple.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            lr: Learning rate.
        """
        # TODO: Implement the training of this model.
        total_train = len(training_data)
        total_val = len(val_data)
        total_data = training_data + val_data
        for i in range(num_epochs):
            random.shuffle(total_data)
            train = total_data[:total_train]
            val = total_data[total_train:]
            correct = 0
            val_correct = 0
            for data, label in train:
                pred = self.predict(data)
                if self.mode:
                    label = self.convert_class(label)
                if pred == int(label):
                    correct += 1
                self.update_parameters(data, pred, int(label), lr)
            for data, label in val:
                pred = self.predict(data)
                if self.mode:
                    label = self.convert_class(label)
                if pred == int(label):
                    val_correct += 1

            if i % 10 == 0:
                print("training accuracy: ", correct / total_train, " validation accuracy: ", val_correct / total_val)

    def add_class_converter(self):
        self.mode = 1
        self.class_dict = {}
        classes = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
                   "comp.windows.x", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey",
                   "sci.crypt", "sci.electronics", "sci.med", "sci.space", "misc.forsale", "talk.politics.misc",
                   "talk.politics.guns", "talk.politics.mideast", "talk.religion.misc", "alt.atheism",
                   "soc.religion.christian"]
        for i in range(self.labels):
            self.class_dict[classes[i]] = i

    def get_mode(self):
        return self.mode

    def convert_class(self, label: str):
        return self.class_dict[label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perceptron model')
    parser.add_argument('-d', '--data', type=str, default='sst2',
                        help='Dataset')
    parser.add_argument('-f', '--features', type=str, default='feature_name', help='Feature type')
    parser.add_argument('-m', '--model', type=str, default='perceptron', help='Model type')
    args = parser.parse_args()

    data_type = args.data
    feature_type = args.features
    model_type = args.model

    ngram = 1
    train_data, val_data, dev_data, test_data = load_data(data_type, feature_type, model_type)

    train_data, val_data, dev_data, test_data, feature_dict = process_data(train_data, val_data, dev_data, test_data)
    num_feature = len(feature_dict)
    num_class = get_num_classes(data_type)

    # Train the model using the training data.
    model = PerceptronModel(num_feature + 2, num_class)
    if data_type == "newsgroups":
        model.add_class_converter()
    print("Training the model...")
    num_epochs = 30
    lr = 0.001
    # Note: ensure you have all the inputs to the arguments.
    model.learn(train_data, val_data, num_epochs, lr)

    # Predict on the development set.
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", f"perceptron_{data_type}_{feature_type}_dev_predictions.csv"))
    print(dev_accuracy)
    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.

    evaluate(model,
             test_data,
             os.path.join("results", f"perceptron_{data_type}_{feature_type}_test_predictions.csv"))
