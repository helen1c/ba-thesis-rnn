import numpy as np
from podium import BucketIterator
from podium import TabularDataset, Vocab, Field
from podium.vectorizers import GloVe
from podium.vocab import UNK, PAD, EOS, BOS

from DenseLayer import DenseLayer
from RNNLayer import RnnLayer
from loss_functions import CrossEntropyLoss
from optimizers import Adam
from LSTMLayer import LSTMLayer
import pickle

data_path_train_csv = '../../dataset/dd_dataset/train/train/train.csv'


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

print("Vocab load!")

vocab_size = len(vocab)
embedding_dim = embeddings.shape[1]
hidden_dim = 200

num_epochs = 500

criterion = CrossEntropyLoss()
optimizer = Adam(0.003)

classifier = RNNClassifier(input_dim=embedding_dim, hidden_dim=hidden_dim, embeddings=embeddings, vocab_size=vocab_size, use_bias=False, number_of_rnn_layers=2)

def get_one_hots(inputs):
    return np.eye(vocab_size)[inputs]

bucket_iterator = BucketIterator(dataset, batch_size=64, bucket_sort_key=instance_length)
inputs = []
outputs = []

for i in range(10):
    for instance in bucket_iterator:
        inputs.append(np.where(instance.text == 3, 0, instance.text)[:, 0:-1])
        outputs.append(get_one_hots(instance.text[:, 1:]))

for i in range(num_epochs):
    cnt = 0
    bucket_iterator = BucketIterator(dataset, batch_size=64, bucket_sort_key=instance_length)
    for instance in bucket_iterator:
        input_batch = np.where(instance.text == 3, 0, instance.text)[:, 0:-1]
        output_b_wo = instance.text[:, 1:]
        output_batch = get_one_hots(output_b_wo)

        out = classifier.forward(input_batch)

        loss = criterion.forward(output_batch, out)
        dedy = criterion.backward(output_batch)
        gradients, model_params = classifier.backward(dedy)

        optimizer.update_parameters(model_params, gradients)
        cnt += 1

        print(f'=================================================')
        print(f'Epoch number={i + 1} | Loss={loss} | Batch number={cnt}')
        print(f'=================================================')

        if cnt >= 1:
            break


#def save_object(obj, filename):
#    with open(filename, 'wb') as output:  # Overwrites any existing file.
#        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
#        print("Model saved!")


# sample usage
#save_object(classifier, 'lstm_layer_1layer_on_test.pkl')
