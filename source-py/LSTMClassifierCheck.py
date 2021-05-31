from LSTMClassifier import LSTMClassifier
from loss_functions import CrossEntropyLoss
from optimizers import Adam
from podium import BucketIterator
import pickle
import numpy as np
from activations import Softmax

dataset_store_path = 'dataset/dataset_whole_train_train_test_split_7_3_50dim.pkl'


def instance_length(instance):
    _, tokenized = instance.text
    return len(tokenized)


def lowercase(raw):
    """Lowercases the input string"""
    return raw.lower()


class RemoveBlanks:
    def __call__(self, raw, tokenized):
        """Remove punctuation from tokenized data"""
        return raw, [tok for tok in tokenized if tok not in [' ', "\n", "\t"]]


with open(dataset_store_path, 'rb') as infile:
    dataset_train, dataset_test, embeddings, vocab = pickle.load(infile)

vocab_size = len(vocab)
embedding_dim = embeddings.shape[1]
hidden_dim = 150
num_epochs = 100

criterion = CrossEntropyLoss()
optimizer = Adam(0.006)

classifier = LSTMClassifier(input_dim=embedding_dim, hidden_dim=hidden_dim, embeddings=embeddings, vocab_size=vocab_size, use_bias=True)


def get_one_hots(inputs):
    return np.eye(vocab_size)[inputs]


bucket_iter_test = BucketIterator(dataset_test, batch_size=64, bucket_sort_key=instance_length)
bucket_iterator = BucketIterator(dataset_train, batch_size=64, bucket_sort_key=instance_length, shuffle=True)

# for instance in bucket_iterator:
#    inputs.append(np.where(instance.text == 3, 0, instance.text)[:, 0:-1])
#    outputs.append(get_one_hots(instance.text[:, 1:]))

# for instance in bucket_iter_test:
#    test_inputs.append(np.where(instance.text == 3, 0, instance.text)[:, 0:-1])
#    test_outputs.append(instance.text[:, 1:])

print("Iterating end.")


def validate(test_output, predictions, n):
    cnt = 0
    for k in range(predictions.shape[0]):
        for l in range(predictions.shape[1]):
            one_hot = predictions[k, l, :]
            indices = (-one_hot).argsort()[:n]
            outp = test_output[k, l]
            if outp in indices:
                cnt += 1
    return (cnt * 1.0) / (test_output.shape[0] * test_output.shape[1])


topN = 10

for i in range(num_epochs):
    bn=0
    for instance in bucket_iterator:
        out = classifier.forward(np.where(instance.text == 3, 0, instance.text)[:, 0:-1])
        one_hots = get_one_hots(instance.text[:, 1:])
        loss = criterion.forward(one_hots, out)
        dedy = criterion.backward(one_hots)
        gradients, model_params = classifier.backward(dedy)
        optimizer.update_parameters(model_params, gradients)
        bn+=1
        print(f'Epoch number={i + 1} | Loss={loss} | Batch number={bn}')

    sum_prec = 0.
    cnt = 0.
    for instance in bucket_iter_test:
        predictions = Softmax.forward(classifier.forward(np.where(instance.text == 3, 0, instance.text)[:, 0:-1]))

        prec = validate(instance.text[:, 1:], predictions, topN)
        sum_prec += prec
        cnt += 1.
        print(f'Precision on {cnt}. batch --> P @ {topN} = {prec}')
    print(f'Average precision after {i + 1} epochs on {cnt} batches -> P @ {topN} = {sum_prec / (cnt * 1.)}')

    if i % 5 == 0 and i > 0:
        model_save = f'models/model_1_l_lstm_train_whole_50d_epoch_={i + 1}.pkl'
        with open(model_save, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(classifier, output, pickle.HIGHEST_PROTOCOL)
            print(f'Model saved, epoch num = {i + 1}')

model_save_path = 'models/rnn_2l_7_3_test_final_2layers.pkl'
with open(model_save_path, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(classifier, output, pickle.HIGHEST_PROTOCOL)
    print("Model saved!")
