import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from data_toolbox.nn_layers import Residual as NNResidual, Dropout


# XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
Y = np.array([
    0,
    1,
    1,
    0,
])


def run_model(use_skip_connections=True):
    if not use_skip_connections:
        def Residual(x): return x
    else:
        Residual = NNResidual
    inp = Input((2,))
    out = inp
    out = Dense(8, activation='tanh')(out)
    out = Residual(Dropout(Dense(8, activation='tanh')))(out)
    out = Residual(Dropout(Dense(8, activation='tanh')))(out)
    out = Residual(Dropout(Dense(8, activation='tanh')))(out)
    out = Residual(Dropout(Dense(8, activation='tanh')))(out)
    out = Residual(Dropout(Dense(8, activation='tanh')))(out)
    out = Residual(Dropout(Dense(8, activation='tanh')))(out)
    out = Residual(Dropout(Dense(8, activation='tanh')))(out)
    out = Residual(Dropout(Dense(8, activation='tanh')))(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(inp, out)
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(X, Y, epochs=500, verbose=False)
    print(model.predict(X))


print('XOR training with a deep dense network')
print('- Groundtruth')
print(Y)

print('- Demo 1: 500 epochs without skip connections')
run_model(use_skip_connections=False)

print('- Demo 2: 500 epochs with skip connections')
run_model(use_skip_connections=True)

"""
XOR training with a deep dense network
- Groundtruth
[0 1 1 0]
- Demo 1: 500 epochs without skip connections
[[0.14449154]
 [0.7745809 ]
 [0.7763253 ]
 [0.7739547 ]]
- Demo 2: 500 epochs with skip connections
[[0.04051533]
 [0.9690992 ]
 [0.9426614 ]
 [0.057626  ]]
"""
