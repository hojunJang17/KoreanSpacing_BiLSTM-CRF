import re
import torch
import pickle
import json
from model.net import BiLSTM_CRF


def result(x, y):
    """
    takes x, y and converts into string
    :param x: token indices
    :param y: label indices
    :return: resulting string
    """
    A = ''
    for i in range(len(x)):
        A += x[i]
        if y[i] == '<split>':
            A += ' '
    return A


with open('experiment/config.json') as f:
    params = json.loads(f.read())


# loading vocabs (token_vocab, label_vocab)
token_vocab_path = params['filepath'].get('token_vocab')
label_vocab_path = params['filepath'].get('label_vocab')
with open(token_vocab_path, 'rb') as f:
    token_vocab = pickle.load(f)
with open(label_vocab_path, 'rb') as f:
    label_vocab = pickle.load(f)


# loading trained model
save_path = params['filepath'].get('ckpt')
ckpt = torch.load(save_path)
hidden_size = params['model'].get('lstm_hidden_size')
model = BiLSTM_CRF(label_vocab, token_vocab, hidden_size)
model.load_state_dict(ckpt['model_state_dict'])


while True:
    original_text = input("Input:\t")
    if original_text == '':
        print('exit')
        break
    treg = re.compile('<\w*>|</\w*>')
    wreg = re.compile(' ')
    a = [treg.sub('', str(t)).strip() for t in original_text]
    a = [t.lower() for t in a if t]
    a = [wreg.sub('', str(t)) for t in a]

    b = torch.tensor(token_vocab.to_indices(a))
    b = b.view(1, -1)

    _, yhat = model(b)

    print('Output:\t', result(a, label_vocab.to_tokens(yhat[0])), '\n')