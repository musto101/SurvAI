from sksurv.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import the class
from SurvAI.src.survai.trees.randomForest import SurvRandomForest

X, y = load_breast_cancer()
Xt = OneHotEncoder(sparse_output=False).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2)

# assert that the model is initialized correctly
assert SurvRandomForest().n_estimators == 100
assert SurvRandomForest().min_samples_split == 2
assert SurvRandomForest().min_samples_leaf == 1
assert SurvRandomForest().max_features == 'sqrt'
assert SurvRandomForest().n_jobs is None

# assert that the model is fitted correctly
model = SurvRandomForest()
model.fit(X_train, y_train)
assert model.model is not None

# assert that the model is predicted correctly
y_pred = model.predict(X_test)
assert y_pred.any() is not None

# assert that the model is scored correctly
score = model.score(X_test, y_test)
assert score is not None

# assert that the model will take in the correct parameters
model = SurvRandomForest(n_estimators=200, min_samples_split=3, min_samples_leaf=2, max_features='log2', n_jobs=2)
assert model.n_estimators == 200
assert model.min_samples_split == 3
assert model.min_samples_leaf == 2
assert model.max_features == 'log2'
assert model.n_jobs == 2

# assert that the model will return the correct string representation
model = SurvRandomForest(n_estimators=200, min_samples_split=3, min_samples_leaf=2, max_features='log2', n_jobs=2)
assert model.__repr__() == ("RandomSurvivalForest(n_estimators=200, min_samples_split=3, min_samples_leaf=2, "
                            "max_features=log2, n_jobs=2, random_state=None)")

# assert that the model will return the correct concordance index
c_index = model.concordance_index(X_test, y_test)
assert c_index is not None
