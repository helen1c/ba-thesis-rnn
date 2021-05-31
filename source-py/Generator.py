from LSTMClassifier import LSTMClassifier
import pickle
import numpy as np
from activations import Softmax
from RNNClassifier import RNNClassifier
from LSTMLayer import LSTMLayer
from activations import Tanh
from activations import Sigmoid
from DenseLayer import DenseLayer

model_path = 'models/final2.pkl'

with open(model_path, 'rb') as infile:
    classifier = pickle.load(infile)


def get_indices_next_w(predictions):
    words = []
    for k in range(predictions.shape[0]):
        for l in range(predictions.shape[1]):
            one_hot = predictions[k, l, :]
            indices = (-one_hot).argsort()[:10]
            words.append(indices)
    return words

def instance_length(instance):
    _, tokenized = instance.text
    return len(tokenized)


def truncate(raw, tokenized, max_length=120):
    return raw, tokenized[:max_length]

def lowercase(raw):
    """Lowercases the input string"""
    return raw.lower()


class RemoveBlanks:
    def __call__(self, raw, tokenized):
        """Remove punctuation from tokenized data"""
        return raw, [tok for tok in tokenized if tok not in [' ', "\n", "\t"]]
dataset_store_path = 'dataset/finalized_dataset.pkl'
with open(dataset_store_path, 'rb') as infile:
    dataset_train, dataset_test, embeddings, vocab = pickle.load(infile)

words = ['imagine']
word = ''
while word != '<EOS>':
    indices = []
    for w in words:
        indices.append(vocab.stoi[w])
    arr = np.array(indices).reshape((1, len(indices)))
    out = classifier.forward(arr)
    predictions = Softmax.forward(out)
    predicted_words = get_indices_next_w(predictions)

    word = vocab.itos[predicted_words[len(words) - 1][np.random.randint(2, size=1)[0]]]
    words.append(word)


print(words)


