import pandas as pd

from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split, cross_val_score

class ScikitDataset:
    def __init__(self, dataset:pd.DataFrame):
        self.scikit_dataset = Bunch()
        self.scikit_dataset['data'] = dataset.drop(columns = 'class')
        self.scikit_dataset['target'] = dataset['class']
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.scikit_dataset['data'], self.scikit_dataset['target'], random_state = 0)
