# 🌇 data-toolbox

data-toolbox is a collection of data science/machine learning utilities for myself. The goal is to automate trivial tasks so I can re-iterate on data projects faster.

This won't be published on PyPI, just provided as is on Github.

```
pip install git+https://github.com/stewartpark/data-toolbox
```


## GloVe Text Vectorizer/Embedding

[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) is a well-known pretrained vector representation for words. Its resulting vectors that are from simliar words get placed nearby in the word vector space, which helps your model understand meanings of words without training your own word to vector representation/embedding.

Training an embedding layer in NLP is expensive, and using pre-trained embedding weights involves a number of tasks to be done in order to plug it into your model. Now it's as simple as below:

```
>>> from data_toolbox.preprocessing.glove import GloVe
>>> glove = GloVe()
[+] Downloading http://nlp.stanford.edu/data/glove.6B.zip...
100%|███████████████████████████████████████| 862M/862M [06:26<00:00, 2.23MiB/s]
400000it [00:12, 30958.43it/s]
>>> sents = ['Beautiful is better than ugly.', 'Explicit is better than implicit.', 'Simple is better than complex.', 'Complex is better than complicated.']

# scikit-learn style fit, transform, fit_transform:
>>> vec = glove.fit_transform(sents)
>>> vec
array([[-0.18173   ,  0.49759001,  0.46325999, ..., -0.23394001,
         0.47297999, -0.028803  ],
       [ 0.4129    , -0.38106999,  0.13273001, ..., -0.23394001,
         0.47297999, -0.028803  ],
       [-0.74169999,  0.78384   , -0.10146   , ..., -0.23394001,
         0.47297999, -0.028803  ],
       [-0.16609   ,  0.36294001,  0.01681   , ..., -0.23394001,
         0.47297999, -0.028803  ]])

# As a keras embedding layer with pretrained GloVe weights:
>>> glove.Embedding()
<tensorflow.python.keras.layers.embeddings.Embedding object at 0x7fd6e63880d0>
```

As you can see, you get to skip a lot of tasks you would normally perform to use pretrained embedding weights such as:

- Downloading the weights and caching it
  - Required files will be downloaded and cached under `~/.cache/data-toolbox/`.
- Building the wordlist/vocabulary from your dataset
  - Accessible by `glove.tokens`
  - You can give it initial or special tokens (`initial_tokens`)
  - It builds the wordlist automatically if you tokenize your texts with `glove.tokenize(..., discover_tokens=True)` or `glove.fit(...)`
- Choosing the maximum size of a vector based on your dataset
  - Accessible by `glove.max_length`
  - You can give it a initial length (`max_length`)
- Based on the above, building your own embedding matrix
- Handling edge cases where certain words exist in your dataset, but not in GloVe
  - Such tokens will be replaced with the `<|UNK|>`(unknown) token when tokenizing
  - You can list all the tokens that became the unknown token by `glove.unknown_tokens`
- Handling special tokens (e.g. in the context of seq2seq, `<|GO|>`, ...)
  - Such tokens will have randomized embedding weights from a normal distribution, instead of just zero-filled weights.
    - The model can distinguish and learn different special tokens.
- Tokenizing texts based on GloVe's word list
  - If you use other tokenizers or your own, you will see a higher volume of unknown tokens, since the way you tokenize can differ from how GloVe was tokenized and trained.
  - The built-in tokenizer will use GloVe's wordlist to tokenize in order to minimize unknown tokens.
  - `glove.tokenize(...)`
- Vectorizing texts via one-hot encoding out of your wordlist indices
  - `glove.to_onehot(...)`
  - Or, `glove.vectorize(..., to_onehot=True)`
- Integrating with your model
  - for scikit-learn, `glove.fit_transform(...)`
  - for keras, `glove.Embedding(...)`
- Persisting the vectorizer
  - To save, `glove.save('filename.pkl')`
  - To load, `glove = GloVe.load('filename.pkl')`
- ...

You can choose to use one of the other bigger GloVe weights like below:

```
>>> glove = GloVe(300, '42B.300d')
[+] Downloading http://nlp.stanford.edu/data/glove.42B.300d.zip...
...
```

## Keras Utility Layers

- `Residual(layer, n_outputs=None, choose=0, normalization_layer_class=tf.keras.layers.LayerNormalization, normalization_layer_options={})`
  - If you wrap a layer with this, it wraps a skip connection around it
- `Dropout(layer, rate=0.2)`
  - If you wrap a layer with this, it adds a drop out layer after the layer specified.
- `MultiHead(layer, n_heads)`
  - If you wrap a layer with this, you can duplicate multiple layers that attach to the same parent.
  - `layer` can be a list of layers with different configurations. In such case, `n_heads` doesn't need to be specified.
- Examples
  - [deep_xor.py](https://github.com/stewartpark/data-toolbox/blob/master/examples/deep_xor.py) (Residual, Dropout)
  - [multihead.py](https://github.com/stewartpark/data-toolbox/blob/master/examples/multihead.py) (MultiHead)
 
