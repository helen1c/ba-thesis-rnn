from podium import TabularDataset, Vocab, Field
from podium.vectorizers import GloVe
from podium import BucketIterator
from podium.vocab import UNK, PAD, EOS, BOS
import numpy as np

data_path_train_csv = '../dataset/dd_dataset/train/train/train.csv'


def lowercase(raw):
    """Lowercases the input string"""
    return raw.lower()


class RemoveBlanks:
    def __call__(self, raw, tokenized):
        """Remove punctuation from tokenized data"""
        return raw, [tok for tok in tokenized if tok not in [' ', "\n", "\t"]]


vocab = Vocab(max_size=10000, min_freq=2, specials=(PAD(), UNK(), BOS(), EOS()))
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

print(f"For vocabulary of size: {len(vocab)} loaded embedding matrix of shape: {embeddings.shape}")


def instance_length(instance):
    _, tokenized = instance.text
    return len(tokenized)


bucket_iterator = BucketIterator(dataset, batch_size=32, bucket_sort_key=instance_length)


def get_embeddings(batch, embeddings):
    w2vec = np.zeros((batch.shape[0], batch.shape[1], embeddings.shape[1]))
    for i in range(batch.shape[0]):
        for j in range(batch.shape[1]):
            w2vec[i, j, :] = embeddings[batch[i, j]]

    return w2vec


for iterator in bucket_iterator:
    batch = iterator.text
    print(get_embeddings(batch, embeddings).shape)
    break
