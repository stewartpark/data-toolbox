from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
import numpy as np

from data_toolbox.nn_layers import MultiHead


def f(x):
    return 2 * x + 2


X = np.linspace(0, 100)
Y = np.array([f(x) for x in X])

inp = Input((1,))
out = inp
out = MultiHead(Dense(1, activation='linear'), n_heads=10)(
    out)  # returns 10 outputs from each dense
out = Concatenate()(out)  # concatenate all these outputs for demo
out = Dense(1, activation='linear')(out)
model = Model(inp, out)
model.summary()
model.compile(loss='mse', optimizer='rmsprop')
model.fit(X, Y, epochs=500)

Yp = model.predict(X)
for i in range(10):
    print(f'f({X[i]}) = {Y[i]}, pred: {Yp[i]}')

# you can wrap MultiHead(...) on a layer and the output will be a list of outputs.
