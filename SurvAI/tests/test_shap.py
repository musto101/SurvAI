from sksurv.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import the shap class
from SurvAI.src.survai.shap.shap import shap
from SurvAI.src.survai.linear.elasticNet import SurvElasticNet

X, y = load_breast_cancer()
Xt = OneHotEncoder(sparse_output=False).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2)

# load and fit the model
model = SurvElasticNet()
model.fit(X_train, y_train)

# create the shap object
shap = shap(model, X_train, y_train, X_test, y_test)
# assert that the shap object is initialized correctly
assert shap.model == model

# assert that the shap object is fitted correctly
shap.fit()
assert shap.explainer != None
assert shap.model_shap != None

# assert that the shap object is predicted correctly
shap.predict()
assert shap.predict_shap != None

print("All tests passed!")