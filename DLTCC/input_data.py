from __future__ import absolute_import

import numpy as np
import collections


Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):

    def __init__(self, data, target):

        assert data.shape[0] == target.shape[0], (
            'data.shape: %s target.shape: %s' %(data.shape, target.shape)
        )
        self._num_examples = data.shape[0]

        self._data = data
        self._target = target
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    def next_batch(self, batch_size, shuffle=True):

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._target = self.target[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished the epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            target_rest_part = self._target[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self.data[perm]
                self._target = self.target[perm]

            # start the next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            target_new_part = self._target[start:end]

            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((target_rest_part, target_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._target[start:end]


# load data from npy files. train_dir is a dictionary.
# train_dir is a dict {"train":{"data":"","target":""}, "test":{"data":"","target":""}}
def read_data_sets(train_dir, one_hot=False, reshape=True, validation_size=50):
    if train_dir is None:
        return None

    # train data files
    train_data_dir = train_dir["train"]["data"]
    train_target_dir = train_dir["train"]["target"]

    test_data_dir = train_dir["test"]["data"]
    test_target_dir = train_dir["test"]["target"]

    all_train_data = np.load(train_data_dir)
    all_train_target = np.load(train_target_dir)

    assert len(all_train_data) == len(all_train_target)

    if validation_size <= 0 or validation_size >= len(all_train_data):
        return None

    # train data and target
    train_data = all_train_data[validation_size:]
    train_target = all_train_target[validation_size:]
    train = DataSet(data=train_data, target=train_target)

    # validation data and target
    validation_data = all_train_data[:validation_size]
    validation_target = all_train_target[:validation_size]
    validation = DataSet(data=validation_data, target=validation_target)

    # test data sets
    test_data = np.load(test_data_dir)
    test_target = np.load(test_target_dir)
    test = DataSet(data=test_data, target=test_target)

    return Datasets(train=train, validation=validation, test=test)




















