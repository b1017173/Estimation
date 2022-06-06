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

        scores = []
        for _ in range(30):
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.4)
            # model = RidgeCV()
            model = GaussianNB()
            # model = svm.SVC(max_iter=10000)
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))

        eval = float(np.array(scores).mean())
        return eval
