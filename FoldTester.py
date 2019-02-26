import numpy as np
from itertools import compress

# Import the Classifier
from Logistic import LogisticClassifier
from LDA import LDAClassifier
from DecisionTree import DecisionTree

# Path to the data set
# DATA_SET = "./Data/winequality-red.csv"
# DATA_SET = "./Data/watermelon3.txt"
DATA_SET = "./Data/watermelon2.txt"
FOLD_NUM = 10
ITERATIONS = 50
CLASSIFIER = DecisionTree
CLASSIFIER_PARAMETERS = {
    "Discrete": False
}


def load_csv(file_name, shuffle=True):
    try:
        _tmp = np.loadtxt(file_name, dtype=np.str, delimiter=';')
    except Exception:
        print("File Load Failed.")
        return None
    _dim_name = _tmp[0, :-1]
    _score_name = _tmp[0, -1]
    _dim_name = list(map(lambda s: s.strip(' \"'), _dim_name))
    _score_name = list(map(lambda s: s.strip(' \"'), _score_name))
    _tmp = _tmp[1:, :]
    if shuffle:
        np.random.shuffle(_tmp)
    _data = _tmp[:, 0:-1].astype(np.float)
    _label = _tmp[:, -1].astype(np.float).reshape((-1, 1))
    return {
        'Dim Name': _dim_name,
        'Score Name': _score_name,
        'Data': _data,
        'Label': _label
    }


def data_set_div(data, labels):
    _label_set = set(labels.reshape(-1))
    ret = {}
    for _label in _label_set:
        elem = data[labels.reshape(-1) == _label, :]
        ret[_label] = np.array_split(elem, FOLD_NUM)
    return ret


data_set = load_csv(DATA_SET)
dim_name, score_name, data, label = data_set['Dim Name'], data_set['Score Name'], data_set['Data'], data_set['Label']
data_set_dict = data_set_div(data, label)
data_set_size, dims = data.shape
print("Data set size:", data_set_size)

# Store the precision data for each fold
total_precision_train = []
total_precision_test = []

# Use "one vs the rest" strategy for multi-classification
# classifier_dict = {l: LogisticClassifier(dims) for l in data_set_dict.keys()}
CLASSIFIER_PARAMETERS['Dimension'] = dims
classifier_dict = {l: CLASSIFIER(CLASSIFIER_PARAMETERS) for l in data_set_dict.keys()}

for fold_cnt in range(FOLD_NUM):
    # Generate the train data/label and the test data/label
    train_sel = [i != fold_cnt for i in range(FOLD_NUM)]
    test_sel = [i == fold_cnt for i in range(FOLD_NUM)]
    train_data_fold = None
    train_label_fold = None
    test_data_fold = None
    test_label_fold = None
    predict_test_label = None
    predict_test_value = None
    predict_train_label = None
    predict_train_value = None
    for temp_label, temp_data in data_set_dict.items():
        temp_train_data = np.concatenate(list(compress(temp_data, train_sel)), axis=0)
        temp_test_data = np.concatenate(list(compress(temp_data, test_sel)), axis=0)
        temp_train_label = np.full((temp_train_data.shape[0], 1), temp_label)
        temp_test_label = np.full((temp_test_data.shape[0], 1), temp_label)
        if train_data_fold is None:
            train_data_fold = temp_train_data
            test_data_fold = temp_test_data
            train_label_fold = temp_train_label
            test_label_fold = temp_test_label
        else:
            train_data_fold = np.concatenate((train_data_fold, temp_train_data), axis=0)
            train_label_fold = np.concatenate((train_label_fold, temp_train_label), axis=0)
            test_data_fold = np.concatenate((test_data_fold, temp_test_data), axis=0)
            test_label_fold = np.concatenate((test_label_fold, temp_test_label), axis=0)
    train_data_num = train_label_fold.shape[0]
    test_data_num = test_label_fold.shape[0]

    # Shuffle the train data
    train_pair = np.concatenate((train_data_fold, train_label_fold), axis=1)
    np.random.shuffle(train_pair)
    train_data_fold = train_pair[:, :-1].reshape(-1, dims)
    train_label_fold = train_pair[:, -1].reshape(-1, 1)

    # Train and test each classifier
    for classifier_positive_label in classifier_dict.keys():
        # Generate the label for the OvR classifier
        classifier_train_label = train_label_fold == classifier_positive_label
        classifier = classifier_dict[classifier_positive_label]
        classifier.fit(train_data_fold, classifier_train_label, iterations=ITERATIONS)

        # Store the prediction
        classifier_predict = classifier.predict(train_data_fold)
        if predict_train_value is None:
            predict_train_value = classifier_predict
            predict_train_label = np.reshape(classifier_positive_label, (1, 1))
        else:
            predict_train_value = np.concatenate(
                (predict_train_value,
                 classifier_predict), axis=1)
            predict_train_label = np.concatenate(
                (predict_train_label,
                 np.reshape(classifier_positive_label, (1, 1))), axis=1)

        classifier_predict = classifier.predict(test_data_fold)
        if predict_test_value is None:
            predict_test_value = classifier_predict
            predict_test_label = np.reshape(classifier_positive_label, (1, 1))
        else:
            predict_test_value = np.concatenate(
                (predict_test_value,
                 classifier_predict), axis=1)
            predict_test_label = np.concatenate(
                (predict_test_label,
                 np.reshape(classifier_positive_label, (1, 1))), axis=1)

    # Get the OvR predication
    predict_train = predict_train_label[:, predict_train_value.argmax(axis=1)]
    predict_test = predict_test_label[:, predict_test_value.argmax(axis=1)]

    # Count
    correct_num_train = (predict_train == train_label_fold.reshape((-1,))).sum()
    correct_num_test = (predict_test == test_label_fold.reshape(-1,)).sum()
    precision_test = correct_num_test / test_data_num
    precision_train = correct_num_train / train_data_num
    print("Fold:", fold_cnt)
    if train_data_num > 0:
        total_precision_train.append(precision_train)
        print("Train Precision:", round(precision_train * 100, 2))
    else:
        print("No Train Data")
    if test_data_num > 0:
        total_precision_test.append(precision_test)
    else:
        print("No Test Data")
        print("Test Precision:", round(precision_test * 100, 2))

total_precision_train = np.average(total_precision_train)
total_precision_test = np.average(total_precision_test)
print("Total Train Precision:", round(total_precision_train * 100, 2))
print("Total Test Precision:", round(total_precision_test * 100, 2))


