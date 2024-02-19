import read_csv
import random
def newsgroups_featurize(train_data, val_data, dev_data, test_data, feature_type):
    """ Featurizes an input for the newsgroups domain.

    Inputs:
        train_data: The training data.
        val_data: The validation data.
        dev_data: The development data.
        test_data: The test data.
        feature_type: Type of feature to be used.
    """
    # TODO: Implement featurization of input.
    pass


def newsgroups_data_loader(train_data_filename: str,
                           train_labels_filename: str,
                           dev_data_filename: str,
                           dev_labels_filename: str,
                           test_data_filename: str,
                           feature_type: str,
                           model_type: str):
    """ Loads the data.

    Inputs:
        train_data_filename: The filename of the training data.
        train_labels_filename: The filename of the training labels.
        dev_data_filename: The filename of the development data.
        dev_labels_filename: The filename of the development labels.
        test_data_filename: The filename of the test data.
        feature_type: The type of features to use.
        model_type: The type of model to use.

    Returns:
        Training, validation, dev, and test data, all represented as a list of (input, data_id, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.
    train_data = read_csv.read(train_data_filename)
    train_label = read_csv.read(train_labels_filename)
    train = list(zip(train_data[1:], train_label[1:]))
    random.shuffle(train)
    n = len(list(train)) // 20
    val = train[:n]
    train = train[n:]
    dev_data = read_csv.read(dev_data_filename)
    dev_label = read_csv.read(dev_labels_filename)
    dev = zip(dev_data[1:], dev_label[1:])

    test_data = read_csv.read(test_data_filename)
    test_label = []
    for i in range(len(test_data)):
        test_label.append(train_label[1])
    test = zip(test_data[1:], test_label[1:])

    # TODO: Featurize the input data for all three splits.

    return train, val, dev, test