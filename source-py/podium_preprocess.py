from podium import TabularDataset, Vocab, Field
from podium.vectorizers import GloVe
from podium import BucketIterator
from podium.vocab import UNK, PAD, EOS, BOS
import numpy as np
import pickle

data_path_train_csv = '../dataset/finalized.csv'


def lowercase(raw):
    return raw.lower()

def truncate(raw, tokenized, max_length=120):
    return raw, tokenized[:max_length]

class RemoveBlanks:
    def __call__(self, raw, tokenized):
        return raw, [tok for tok in tokenized if tok not in [' ', "\n", "\t"]]


vocab = Vocab(max_size=10000, min_freq=2, specials=(PAD(), UNK(), BOS(), EOS()))
text = Field('text',
             numericalizer=vocab,
             pretokenize_hooks=[lowercase],
             posttokenize_hooks=[RemoveBlanks(), truncate],
             tokenizer='spacy-en_core_web_sm')
fields = {'text': text}

dataset = TabularDataset(data_path_train_csv, format='csv', fields=fields)
dataset_train, dataset_test = dataset.split([80, 20], random_state=42)
dataset_train.finalize_fields()

print(dataset_train[2])
print(dataset_train[5])

vocab = fields['text'].vocab
glove = GloVe(dim=100)
embeddings = glove.load_vocab(vocab)

print(f"For vocabulary of size: {len(vocab)} loaded embedding matrix of shape: {embeddings.shape}")


def instance_length(instance):
    _, tokenized = instance.text
    return len(tokenized)


def get_embeddings(batch, embeddings):
    w2vec = np.zeros((batch.shape[0], batch.shape[1], embeddings.shape[1]))
    for i in range(batch.shape[0]):
        for j in range(batch.shape[1]):
            w2vec[i, j, :] = embeddings[batch[i, j]]

    return w2vec

path = "dataset/finalized_dataset.pkl"

with open(path, 'wb') as output:  # Overwrites any existing file.
    pickle.dump((dataset_train, dataset_test, embeddings, vocab), output, pickle.HIGHEST_PROTOCOL)
    print("Dataset saved!")
