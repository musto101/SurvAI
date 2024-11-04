from sksurv.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import the class
from SurvAI.src.survai.linear.elasticNet import SurvElasticNet

X, y = load_breast_cancer()
Xt = OneHotEncoder(sparse_output=False).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2)

# assert that the model is initialized correctly
assert SurvElasticNet().alphas == [1]
assert SurvElasticNet().l1_ratio == 0.5
assert SurvElasticNet().max_iter == 100
assert SurvElasticNet().model == None

# assert that the model is fitted correctly
model = SurvElasticNet()
model.fit(X_train, y_train)
assert model.model != None

# assert that the model is predicted correctly
y_pred = model.predict(X_test)
assert y_pred.any() != None

# assert that the model is scored correctly
score = model.score(X_test, y_test)
assert score != None

# assert that the model will take in the correct parameters
model = SurvElasticNet(alphas=[0.5], l1_ratio=0.3, max_iter=200)
assert model.alphas == [0.5]
assert model.l1_ratio == 0.3
assert model.max_iter == 200

# assert that the model will return the correct string representation
model = SurvElasticNet(alphas=[0.5], l1_ratio=0.3, max_iter=200)
assert model.__repr__() == "CoxnetSurvivalAnalysis(alphas=[0.5], l1_ratio=0.3, max_iter=200)"