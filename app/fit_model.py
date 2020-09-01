import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle
from ..Notebook.src.bankchurn import *


class Model(object):
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None

    def load_data(self, filename):
        # data = load_iris()

        _, X_train, _, _, y_train, _ = prepare_and_split_data(filename)
        # balance it
        X_train_balanced, y_train_balanced = get_balanced_data(X_train, y_train)

        # set it to the class internal attributes
        self.X = X_train_balanced #data.data
        self.y = y_train_balanced #data.target

    def fit_model(self):
        self.model = GradientBoostingClassifier()
        self.model.fit(self.X, self.y)

    def pickle_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == '__main__':
    m = Model()
    m.load_data('../data/bank_churn.csv')
    m.fit_model()
    m.pickle_model('../model/model.pkl')
