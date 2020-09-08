import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

from data_toolbox.preprocessing.glove import GloVe
from data_toolbox.nn_layers.positional_encoder import PositionalEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, LayerNormalization
from tqdm import tqdm
import numpy as np


glove = GloVe()
sents = [
    glove.tokenize(x, discover_tokens=True, adjust_max_length=True) for x in [
        'This is a great movie.',
        'This great movie is a bad movie.',
        'This movie is a bad ass movie.',
        'There are bad movies, but this movie is not one of them.',
        'There are good movies, but this movie is not one of them.',
        'This is a bad movie.',
        'This movie is not good.',
        'This isn\'t a bad movie.',
        'This bad ass movie is bad.',
        'This movie can\'t be bad.',
    ]
]
glove.compile()

X = np.array([glove.vectorize(sent) for sent in sents])
Y = np.array([
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
])


def run_model(enable_positional_encoding=False):
    inp = Input((X.shape[-1],), dtype='float32')
    out = inp
    out = glove.Embedding(mask_zero=True)(out)
    if enable_positional_encoding:
        out = PositionalEncoder(X.shape[-1])(out)
    out = Flatten()(out)
    out = Dense(16, activation='sigmoid')(out)
    out = LayerNormalization(epsilon=1e-06)(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(inp, out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    hist = model.fit(X[:-2], Y[:-2], epochs=50,
                     validation_data=(X[-2:], Y[-2:]), verbose=False)
    return hist.history['val_acc'][-1]


v1, v2 = [], []
for _ in tqdm(range(10)):
    v1.append(run_model(False))
    v2.append(run_model(True))
print('Without positional encoding: avg(val acc) = ', np.mean(v1))
print('With positional encoding: avg(val acc) = ', np.mean(v2))
"""
Without positional encoding: avg(val acc) =  0.4
With positional encoding: avg(val acc) =  0.85
"""
