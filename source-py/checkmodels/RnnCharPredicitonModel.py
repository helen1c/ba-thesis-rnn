import numpy as np

from DenseLayer import DenseLayer
from RNNLayer import RnnLayer
from loss_functions import CrossEntropyLoss
from LSTMLayer import LSTMLayer
from RNNLayer import RnnLayer
from optimizers import SGD

text = ['hey how are you', 'good i am fine', 'have a nice day']

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set(''.join(text))

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

# Finding the length of the longest string in our data
maxlen = len(max(text, key=len))
# Padding

# A simple loop that loops through the list of sentences and adds a ' ' whitespace until the length of
# the sentence matches the length of the longest sentence
for i in range(len(text)):
    while len(text[i]) < maxlen:
        text[i] += ' '

# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])

    # Remove first character for target sequence
    target_seq.append(text[i][1:])
    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


X = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
T = one_hot_encode(target_seq, dict_size, seq_len, batch_size)

hidden_dim = 30
rnn1 = LSTMLayer(dict_size, hidden_dim, use_bias=False)
rnn2 = LSTMLayer(hidden_dim, hidden_dim, use_bias=False)
dense = DenseLayer(hidden_dim, dict_size, use_bias=False)

clos = CrossEntropyLoss()
n_epochs = 1000
learning_rate = 0.1

from optimizers import Adam

optimizer = Adam(0.03)

for i in range(n_epochs):
    H, _, _ = rnn1.forward(X)
    o = H[:, 1:, :]
    H2, _, _ = rnn2.forward(o)
    out = dense.forward(H2[:, 1:, :])
    loss = clos.forward(T, out)
    if i % 10 == 0:
        print(f'{i + 1}. epoha- loss: {loss}')

    dEdY = clos.backward(T)

    de_dx, de_dw, de_db_d = dense.backward(dEdY)
    dEdW_in1, dEdW_hh1, de_db_r1, grad = rnn2.backward(o, de_dx)
    dEdW_in2, dEdW_hh2, de_db_r2, _ = rnn1.backward(X, grad)

    params = [dense.weights, rnn1.input_weights, rnn1.hidden_weights, rnn2.input_weights, rnn2.hidden_weights]
    update_params = [de_dw, dEdW_in2, dEdW_hh2, dEdW_in1, dEdW_hh1]

    optimizer.update_parameters(params, update_params)
