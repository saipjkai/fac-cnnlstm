import os
import pickle
import numpy as np

from keras.utils import Sequence
from keras.utils import to_categorical


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def load_pkl(pkl_path):
    """
        Loads pickle
    """
    with open(pkl_path, 'rb') as pkl:
        uPkldData = pickle.load(pkl)
    return uPkldData


def get_Xy(data_path, dims, data_flag='train'):
    """
        Get's X, y given data path & it's single unit dimensions
    """
    X_path = os.path.join(data_path, 'X{}_{}x{}x{}x{}.pkl'.format(data_flag, *dims))
    X = load_pkl(X_path) 

    y_path = os.path.join(data_path, 'y{}_{}x{}x{}x{}.pkl'.format(data_flag, *dims))
    y = load_pkl(y_path)
    y = to_categorical(y).astype(np.int32)
    return X, y