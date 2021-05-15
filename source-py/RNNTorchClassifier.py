import torch
from torch import nn
import numpy as np
import numpy as np
from podium import BucketIterator
from podium import TabularDataset, Vocab, Field
from podium.vectorizers import GloVe
from podium.vocab import UNK, PAD, EOS, BOS


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

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

RNN_TYPES = ['RNN', 'LSTM', 'GRU']

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):

        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        rnn_out, _ = self.rnn(X)
        fc_out = self.fc(rnn_out.contigious().view(-1, self.hidden_dim))
        return fc_out


def get_embeddings(mini_batch, embeddings):
    w2vec = np.zeros((mini_batch.shape[0], mini_batch.shape[1], embeddings.shape[1]))
    for m in range(mini_batch.shape[0]):
        for j in range(mini_batch.shape[1]):
            w2vec[m, j, :] = embeddings[mini_batch[m, j]]

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

num_epochs = 500

model = RNN(embedding_dim=embedding_dim, output_dim=vocab_size, hidden_dim=hidden_dim)

model.to(device)

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters())

for i in range(num_epochs):
    cnt = 0
    bucket_iterator = BucketIterator(dataset, batch_size=32, bucket_sort_key=instance_length)
    for instance in bucket_iterator:
        optimizer.zero_grad()
        input_batch = np.where(instance.text == 3, 0, instance.text)[:, 0:-1]
        w2v = get_embeddings(input_batch, embeddings)
        tensor_input = torch.from_numpy(w2v)
        output, hidden = model(tensor_input.to(device))
        output_b_wo = instance.text[:, 1:]

        loss = criterion.forward(output, torch.from_numpy(output_b_wo).view(-1).long())
        loss.backward()
        optimizer.step()


        print(f'Epoha={i + 1}. loss={loss.item} minibatch_number={cnt + 1}')

        cnt += 1



