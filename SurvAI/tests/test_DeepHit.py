from sksurv.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# import the deephit class
from SurvAI.src.survai.nn.DeepHit import DeepHit

X, y = load_breast_cancer()
Xt = OneHotEncoder(sparse_output=False).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2)

input_dims = {'x_dim': 30, 'num_Event': 3, 'num_Category': 10}
network_settings = {'h_dim_shared': 128, 'h_dim_CS': 64, 'num_layers_shared': 3, 'num_layers_CS': 2,
                    'active_fn': 'relu', 'initial_W': 'xavier'}

# assert that the model is initialized correctly
model = DeepHit(input_dims, network_settings)
assert model.x_dim == 30
assert model.num_Event == 3
assert model.num_Category == 10
assert model.h_dim_shared == 128
assert model.h_dim_CS == 64
assert model.num_layers_shared == 3
assert model.num_layers_CS == 2
assert model.active_fn == 'relu'
assert model.initial_W == 'xavier'
assert model.output_layer != None
assert model.h_dim_shared == 128
assert model.h_dim_CS == 64
assert model.num_layers_shared == 3
assert model.num_layers_CS == 2

print("All tests passed!")

