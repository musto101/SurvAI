import pandas as pd
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.preprocessing import OneHotEncoder

class SurvElasticNet:
    def __init__(self, alphas=[1], l1_ratio=0.5, max_iter=100, random_state=None):
        self.model = None
        self.alphas = alphas
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def fit(self, X, y):
        self.model = CoxnetSurvivalAnalysis(alphas=self.alphas, l1_ratio=self.l1_ratio, max_iter=self.max_iter)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def concordance_index(self, X, y):
        return self.model.score(X, y)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)

    def __repr__(self):
        return (f"CoxnetSurvivalAnalysis(alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
                f"max_iter={self.max_iter}, random_state={self.random_state})")


# example of model initialization
model = SurvElasticNet()

# # example of model fitting
# from sksurv.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
#
#
# X, y = load_breast_cancer()
# Xt = OneHotEncoder(sparse_output=False).fit_transform(X)
#
# X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2)
#
# model.fit(X_train, y_train)
#
# # example of model prediction
# y_pred = model.predict(X_test)
#
# # example of model scoring
# score = model.score(X_test, y_test)
