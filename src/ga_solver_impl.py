from turtle import shape
import numpy as np

from sklearn import svm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from ga import GaSolver

class GaSolverImpl(GaSolver):

    # override
    def evaluate_individual(self, individual, X, y):
        use_cols = [bool(gene) for gene in individual.chromosome]
        X_temp = X.iloc[:, use_cols]

        test_size = int(len(X_temp) * 0.3)
        train_data = X_temp[test_size:]
        test_data = X_temp[:test_size]
        train_label = y[test_size:]
        test_label = y[:test_size]
        # print(train_data.shape)
        # print(test_data.shape)
        # X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.4)
        # model = RidgeCV()
        model = GaussianNB()
        # model = svm.SVC(max_iter=10000)
        # model.fit(X_train, y_train)
        model.fit(train_data, train_label)
        # print(model.score(test_data, test_label))

        return model.score(test_data, test_label)
        # print(model.score(X_test, y_test))
        # return model.score(X_test, y_test)

        feature_count_evaluation = 1 - (train_data.shape[1] / self.chromosome_length)

        return (model.score(test_data, test_label) + feature_count_evaluation) / 2.0
