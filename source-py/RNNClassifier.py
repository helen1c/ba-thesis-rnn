import numpy as np
from podium import BucketIterator
from podium import TabularDataset, Vocab, Field
from podium.vectorizers import GloVe
from podium.vocab import UNK, PAD, EOS, BOS

from DenseLayer import DenseLayer
from RNNLayer import RnnLayer
from loss_functions import CrossEntropyLoss
from optimizers import Adam

data_path_train_csv = '../dataset/dd_dataset/test/test/test.csv'


def lowercase(raw):
    """Lowercases the input string"""
    return raw.lower()


class RemoveBlanks:
    def __call__(self, raw, tokenized):
        """Remove punctuation from tokenized data"""
        return raw, [tok for tok in tokenized if tok not in [' ', "\n", "\t"]]


def instance_length(instance):
    _, tokenized = instance.text
    return len(tokenized)


class RNNClassifier(object):
    def __init__(self, input_dim, hidden_dim, vocab_size, embeddings, number_of_rnn_layers=1, use_bias=True):
        self.embedding_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.use_bias = use_bias

        self.rnn_layer_0 = RnnLayer(input_dim, hidden_dim, use_bias=False)
        self.rnn_layer_1 = RnnLayer(hidden_dim, hidden_dim, use_bias=False)
        self.dense = DenseLayer(hidden_dim, vocab_size, use_bias=False)

        self.inputs = None

        self.H_0 = None
        self.H_1 = None

    def forward(self, inputs):

        self.inputs = self.get_embeddings(inputs)
        self.H_0, _ = self.rnn_layer_0.forward(self.inputs)
        self.H_1, _ = self.rnn_layer_1.forward(self.H_0[:, 1:, :])
        return self.dense.forward(self.H_1[:, 1:, :])

    def backward(self, dEdY):
        dense_x, dense_w, dense_b = self.dense.backward(dEdY)
        rnn_1_wih, rnn_1_whh, rnn_1_b, rnn_1_x = self.rnn_layer_1.backward(self.H_0[:, 1:, :], self.H_1, dense_x)
        rnn_2_wih, rnn_2_whh, rnn_2_b, _ = self.rnn_layer_0.backward(self.inputs, self.H_0, rnn_1_x)

        return [dense_w, rnn_1_wih, rnn_1_whh, rnn_2_whh, rnn_2_wih], [self.dense.weights, self.rnn_layer_1.input_weights, self.rnn_layer_1.hidden_weights, self.rnn_layer_0.hidden_weights,
                                                                       self.rnn_layer_0.input_weights]

    def get_embeddings(self, mini_batch):
        w2vec = np.zeros((mini_batch.shape[0], mini_batch.shape[1], self.embeddings.shape[1]))
        for m in range(mini_batch.shape[0]):
            for j in range(mini_batch.shape[1]):
                w2vec[m, j, :] = self.embeddings[mini_batch[m, j]]

        return w2vec


vocab = Vocab(max_size=5000, min_freq=2, specials=(PAD(), UNK(), BOS(), EOS()))
text = Field('text',
             numericalizer=vocab,
             pretokenize_hooks=[lowercase],
             posttokenize_hooks=[RemoveBlanks()],
             tokenizer='spacy-en_core_web_sm')
fields = {'text': text}

dataset = TabularDataset(data_path_train_csv, format='csv', fields=fields)
dataset.finalize_fields()

vocab = fields['text'].vocab
glove = GloVe()
embeddings = glove.load_vocab(vocab)

bucket_iterator = BucketIterator(dataset, batch_size=32, bucket_sort_key=instance_length)

vocab_size = len(vocab)
embedding_dim = embeddings.shape[1]
hidden_dim = 150

num_epochs = 300

criterion = CrossEntropyLoss()
optimizer = Adam(0.003)

classifier = RNNClassifier(input_dim=embedding_dim, hidden_dim=hidden_dim, embeddings=embeddings, vocab_size=vocab_size, use_bias=False, number_of_rnn_layers=2)


def get_one_hots(inputs):
    return np.eye(vocab_size)[inputs]


for i in range(num_epochs):
    cnt = 0
    bucket_iterator = BucketIterator(dataset, batch_size=32, bucket_sort_key=instance_length)
    for instance in bucket_iterator:
        input_batch = np.where(instance.text == 3, 0, instance.text)[:, 0:-1]
        output_b_wo = instance.text[:, 1:]
        output_batch = get_one_hots(output_b_wo)

        out = classifier.forward(input_batch)

        loss = criterion.forward(output_batch, out)

        if cnt % 30 == 0:
            print(f'Epoha={i + 1}. loss={loss} minibatch_number={cnt}')
        dedy = criterion.backward(output_batch)
        gradients, model_params = classifier.backward(dedy)

        optimizer.update_parameters(model_params, gradients)
        cnt += 1
