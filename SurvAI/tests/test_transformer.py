from sksurv.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import the class
from SurvAI.src.survai.transformers import transformer

X, y = load_breast_cancer()
Xt = OneHotEncoder(sparse_output=False).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2)

# assert that the model is initialized correctly
trans = transformer.Transformer(features=X, labels=y, num_features=X.shape[1])
assert

# assert that the model is fitted correctly


