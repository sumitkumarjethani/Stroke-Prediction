# %% [code]
# Common functions for training and evaluating models

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Datasets functions
def get_train_test_data():
    train = pd.read_csv('/kaggle/input/oversampling-smote/train_oversampled.csv')
    test = pd.read_csv('/kaggle/input/oversampling-smote/test.csv')
    X_train , y_train = train.drop(columns=['stroke']), train.stroke
    X_test , y_test = test.drop(columns=['stroke']), test.stroke
    return (X_train, X_test, y_train, y_test)

def get_stratified_shuffle_cv(n_splits=5, test_size=0.2, train_size=0.8, seed=99):
    return StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                  train_size=train_size, random_state=seed)

# Model evaluation functions
def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_true=y_test, y_pred=y_pred)

def print_metrics(y_test, y_pred, model_name):
    
    print(classification_report(y_test, y_pred))
    
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr,tpr,'b',label="AUC="+str(auc))
    plt.plot([0,1],[0,1],'k--')
    plt.title(model_name)
    plt.grid()
    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)

def plot_tree_feature_importance(features, importances):
    df = pd.DataFrame(list(zip(features, importances)), columns=['feature', 'importance']).sort_values(['importance'], ascending=False)
    plt.figure(figsize=(10,8))
    clrs = ['green' if (x < max(df.importance)) else 'red' for x in df.importance]
    sns.barplot(y=df.feature,x=df.importance,palette=clrs).set(title='Important features')
    plt.show()
