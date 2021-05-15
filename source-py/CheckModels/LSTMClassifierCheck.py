from LSTMClassifier import LSTMClassifier
from loss_functions import CrossEntropyLoss
from optimizers import Adam
from podium import BucketIterator
import pickle
import numpy as np

dataset_store_path = '../dataset/dataset.pkl'
from activations import Softmax


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
    dataset, embeddings, vocab = pickle.load(infile)

vocab_size = len(vocab)
embedding_dim = embeddings.shape[1]
hidden_dim = 256
num_epochs = 30

criterion = CrossEntropyLoss()
optimizer = Adam(0.003)

classifier = LSTMClassifier(input_dim=embedding_dim, hidden_dim=hidden_dim, embeddings=embeddings, vocab_size=vocab_size, use_bias=True)


def get_one_hots(inputs):
    return np.eye(vocab_size)[inputs]


#from podium import Iterator

# iterator = Iterator(dataset, batch_size=64)
bucket_iterator = BucketIterator(dataset, batch_size=32, bucket_sort_key=instance_length)
inputs = []
outputs = []
test_inputs = []
test_outputs = []
c = 0

limit = 40
test_limit = 50

for instance in bucket_iterator:
    if c >= test_limit:
        break
    if c >= limit:
        test_inputs.append(np.where(instance.text == 3, 0, instance.text)[:, 0:-1])
        test_outputs.append(instance.text[:, 1:])
        c += 1
        continue

    inputs.append(np.where(instance.text == 3, 0, instance.text)[:, 0:-1])
    outputs.append(get_one_hots(instance.text[:, 1:]))
    c += 1

for i in range(num_epochs):
    for j in range(len(inputs)):
        out = classifier.forward(inputs[j])

        loss = criterion.forward(outputs[j], out)
        dedy = criterion.backward(outputs[j])
        gradients, model_params = classifier.backward(dedy)

        optimizer.update_parameters(model_params, gradients)
        if j % len(inputs) == 0:
            print(f'Epoch number={i + 1} | Loss={loss} | Batch number={j + 1}')


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


times = test_limit - limit
sum_prec = 0
for i in range(times):
    predictions = Softmax.forward(classifier.forward(test_inputs[i]))
    prec = validate(test_outputs[i], predictions, 5)
    sum_prec += prec
    print(f'Precision on {i + 1}. batch={prec}\n')

print(f'Average precision on {times} batches and {limit} training batches = {sum_prec / (times * 1.)}')
