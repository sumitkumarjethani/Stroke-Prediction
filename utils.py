# Common functions for training and evaluating models

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

seed = 99

def get_train_test_data():
    train = pd.read_csv('/kaggle/input/oversampling-smote/train_oversampled.csv')
    test = pd.read_csv('/kaggle/input/oversampling-smote/test.csv')
    X_train , y_train = train.drop(columns=['stroke']), train.stroke
    X_test , y_test = test.drop(columns=['stroke']), test.stroke
    return (X_train, X_test, y_train, y_test)

def get_stratified_shuffle_cv(n_splits=5, test_size=0.2, train_size=0.8):
    return StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                  train_size=train_size, random_state=seed)