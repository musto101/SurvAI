from lifelines.utils import concordance_index
import xgboost as xgb

# create a wrapper class for the XGB
class SurvXGB:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, objective='survival:cox', booster='gbtree',
                  min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1, n_jobs=None, random_state=None,):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.booster = booster
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state

    def params(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'booster': self.booster,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state
        }

    def fit(self, X, y):
        self.X_db = xgb.DMatrix(X, label=y['last_DX'], weight=y['last_visit'])
        self.model = xgb.train(self.params(), self.X_db)

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
        return (f"XGBRegressor(n_estimators={self.n_estimators}, max_depth={self.max_depth}, "
                f"learning_rate={self.learning_rate}, objective={self.objective}, booster={self.booster}, "
                f"n_jobs={self.n_jobs}, random_state={self.random_state})")
