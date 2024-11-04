
from sksurv.linear_model import CoxnetSurvivalAnalysis

class SurvElasticNet:
    def __init__(self, alphas=[1], l1_ratio=0.5, max_iter=100):
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
        return (f"CoxnetSurvivalAnalysis(alphas={self.alphas}, l1_ratio={self.l1_ratio}, "
                f"max_iter={self.max_iter})")



