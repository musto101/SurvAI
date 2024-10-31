from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest

# create a wrapper class for the RandomSurvivalForest class

class SurvRandomForest:
    def __init__(self, n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', n_jobs=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        self.model = RandomSurvivalForest(n_estimators=self.n_estimators, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                          n_jobs=self.n_jobs, random_state=self.random_state)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def concordance_index(self, X, y):
        return concordance_index(y, self.model.predict(X))

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)

    def __repr__(self):
        return (f"RandomSurvivalForest(n_estimators={self.n_estimators}, min_samples_split={self.min_samples_split}, "
                f"min_samples_leaf={self.min_samples_leaf}, max_features={self.max_features}, n_jobs={self.n_jobs}, random_state={self.random_state})")



