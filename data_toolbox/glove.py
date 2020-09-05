from data_toolbox.data_loader import ZippedDataLoader, PickledDataLoader

from tensorflow.keras.layers import Embedding
from tensorflow.keras import initializers
from tqdm import tqdm
import numpy as np
import pickle


class GloVe:

    def __init__(
        self, embedding_dim=100, dataset='6B',
        custom_zip_url=None, initial_tokens=[],
        unknown_token='<|UNK|>', padding_token='<|PAD|>',
        max_length=None,
    ):
        self.embedding_dim = embedding_dim
        self._initial_tokens = initial_tokens
        self._unknown_token = unknown_token
        self._padding_token = padding_token
        self._initial_max_length = max_length
        self._dataset = dataset
        self._max_token_length = 0
        self.initialize_glove_lookup_table(
            dataset, custom_zip_url or f'http://nlp.stanford.edu/data/glove.{dataset}.zip')
        self.reset()

    def reset(self):
        self.max_length = self._initial_max_length or 0
        self._compiled = False
        self._tokens = []
        self._tokens_map = {}
        self._unknown_tokens = []
        self.add_token(self._padding_token)
        self.add_token(self._unknown_token)
        for t in self._initial_tokens:
            self.add_token(t)

    def initialize_glove_lookup_table(self, dataset, url):
        loader = ZippedDataLoader(f'glove.{dataset}.zip', url)
        cache = PickledDataLoader(f'glove.{dataset}.pkl')
        table = cache.open(f'{self.embedding_dim}d')
        if not table:
            table = {}
            with loader.open(f'glove.{self._dataset}.{self.embedding_dim}d.txt') as f:
                for line in tqdm(f):
                    line = line.decode('utf-8').strip()
                    value = line.split(' ')
                    word = value[0]
                    coef = np.array(value[1:], dtype='float32')
                    table[word] = coef
            cache.save(f'{self.embedding_dim}d', table)
        self._glove_lookup_table = table
        self._max_token_length = max(
            self._max_token_length,
            max([len(w) for w in table.keys()])
        )

    @property
    def tokens(self):
        return self._tokens

    @property
    def unknown_tokens(self):
        return self._unknown_tokens

    def add_token(self, token):
        if self._compiled:
            raise Exception('This GloVe object is already compiled.')
        if len(token) > self._max_token_length:
            self._max_token_length = len(token)
        self._tokens.append(token)
        self._tokens_map[token] = len(self._tokens_map)

    def tokenize(self, text, discover_tokens=False, adjust_max_length=False):
        if self._compiled and (discover_tokens or adjust_max_length):
            raise Exception(
                'Invalid options: this GloVe object is already compiled.')
        ignore_token = [' ', '\n', '\r', '\t']
        tokens, i = [], 0
        while i < len(text):
            if text[i] in ignore_token:
                i += 1
                continue
            for l in reversed(range(1, self._max_token_length + 1)):
                candidate = text[i:i + l]
                if candidate.lower() in self._tokens_map:  # Case: already learned
                    token = candidate.lower()
                elif candidate.lower() in self._glove_lookup_table:  # Case: in GloVe
                    token = candidate.lower()
                elif candidate in self._tokens_map:  # Case: special token
                    token = candidate
                else:
                    continue

                # Missing word in the user wordlist
                if token not in self._tokens_map:
                    if discover_tokens:
                        self.add_token(token)
                    elif not self._compiled:
                        raise Exception(
                            f'Token not in self.tokens: {token}. Try adding .tokenize(..., discover_tokens=True) if the word list is not finalized.')
                    else:
                        # TODO: find the closest word in the user wordlist via cosine similarity.
                        # Otherwise, this means that the word we just observed wasn't part of the train dataset, but it's in GloVe.
                        token = self._unknown_token
                tokens.append(token)
                i += len(token)
                break
            else:
                self._unknown_tokens.append(text[i])
                tokens.append(self._unknown_token)
                i += 1
        if adjust_max_length:
            self.max_length = max(self.max_length, len(tokens))
        return tokens

    def vectorize(self, tokens, length=None, to_onehot=False, to_glove_vector=False):
        if not self._compiled:
            raise Exception('This GloVe object must be compiled to vectorize')
        length = length or self.max_length
        if len(tokens) > length:
            raise Exception(
                f'Given tokens too long: {len(tokens)} > {length}. Specify the size or try adding .tokenize(..., adjust_max_length=True) when tokenizing.')
        v = np.full(shape=(length,), fill_value=0)
        v[:len(tokens)] = np.array([self._tokens.index(token)
                                    for token in tokens])
        if to_onehot:
            return self.to_onehot(v)
        if to_glove_vector:
            return self.to_glove_vector(v)
        return v

    def to_onehot(self, vector):
        if not self._compiled:
            raise Exception('This GloVe object must be compiled to vectorize')
        v = np.zeros((vector.shape[0], len(self._tokens)))
        for i, c in enumerate(vector):
            if c == 0:
                continue  # zero is reserved for padding
            v[i, int(c)] = 1
        return v

    def to_glove_vector(self, vector):
        if not self._compiled:
            raise Exception('This GloVe object must be compiled to vectorize')
        v = np.zeros((vector.shape[0], self.embedding_dim))
        for i, c in enumerate(vector):
            v[i, :] = self._embedding_matrix[c]
        return v

    def from_index(self, ind):
        return self.tokens[ind]

    def compile(self):
        self._embedding_matrix = np.zeros(
            (len(self._tokens), self.embedding_dim))
        for i, token in enumerate(self._tokens):
            if i == 0:
                # Case: padding. we want this embedding weight to be zero-filled.
                continue
            if token in self._glove_lookup_table:
                self._embedding_matrix[i] = self._glove_lookup_table[token]
            else:
                # Case: custom special tokens. Create a somewhat unique value in the feature space.
                np.random.seed(i)
                self._embedding_matrix[i] = np.random.normal(
                    scale=0.6, size=(self.embedding_dim,))
                np.random.seed()
        self._compiled = True

    # START keras helpers
    def Embedding(self, **kwargs):
        if not self._compiled:
            self.compile()
        return Embedding(
            len(self._tokens),
            self.embedding_dim,
            embeddings_initializer=initializers.Constant(
                self._embedding_matrix),
            mask_zero=True,
            trainable=False,
            **kwargs,
        )
    # END

    # START scikit-learn helpers
    def fit_transform(self, texts):
        self.reset()
        ts = [
            self.tokenize(
                text,
                discover_tokens=True,
                adjust_max_length=True
            ) for text in texts
        ]
        self.compile()
        return np.array(
            [self.vectorize(t, to_glove_vector=True) for t in ts]
        ).reshape((len(texts), self.max_length * self.embedding_dim))

    def fit(self, texts):
        self.fit_transform(texts)  # just discard the result

    def transform(self, texts):
        return np.array(
            [self.vectorize(self.tokenize(text), to_glove_vector=True)
             for text in texts]
        ).reshape((len(texts), self.max_length * self.embedding_dim))
    # END

    # START persistence
    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    # END
