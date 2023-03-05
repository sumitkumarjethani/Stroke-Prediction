# Common functions for training and evaluating models

import pandas as pd

def get_train_test_data():
    train = pd.read_csv('/kaggle/input/oversampling-smote/train_oversampled.csv')
    test = pd.read_csv('/kaggle/input/oversampling-smote/test.csv')
    X_train , y_train = train.drop(columns=['stroke']), train.stroke
    X_test , y_test = test.drop(columns=['stroke']), test.stroke
    return (X_train, X_test, y_train, y_test)