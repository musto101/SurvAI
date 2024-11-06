from sksurv.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import the xgboost class
from SurvAI.src.survai.trees.xgboost import SurvXGB
from SurvAI.tests.test_shap import y_train

X, y = load_breast_cancer()
Xt = OneHotEncoder(sparse_output=False).fit_transform(X)

y_train[1]
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2)

# assert that the model is initialized correctly
assert SurvXGB().n_estimators == 100
assert SurvXGB().max_depth == 3
assert SurvXGB().learning_rate == 0.1
assert SurvXGB().objective == 'survival:cox'
assert SurvXGB().booster == 'gbtree'
assert SurvXGB().min_child_weight == 1
assert SurvXGB().gamma == 0
assert SurvXGB().subsample == 1
assert SurvXGB().colsample_bytree == 1
assert SurvXGB().n_jobs == None
assert SurvXGB().random_state == None

# assert that the model is fitted correctly
model = SurvXGB()
model.fit(X_train, y_train)
assert model.model != None
