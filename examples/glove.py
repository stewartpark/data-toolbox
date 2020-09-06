from data_toolbox.preprocessing.glove import GloVe
import sys
sys.path.append('.')


glove = GloVe()
tokens = glove.tokenize(
    'hi, this is an example.',
    discover_tokens=True,
    adjust_max_length=True,
)
print(tokens)

# Freeze the wordlist and build the embedding matrix
glove.compile()

# Now, we can vectorize stuff
vector = glove.vectorize(tokens, length=16)
print(vector)

vector = glove.vectorize(tokens, length=16, to_glove_vector=True)
print(vector)

glove.save('test.pkl')
