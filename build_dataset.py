from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from glob import glob
import pickle
from collections import Counter
from model.utils import Vocab


def get_label(s, idx=0):
    # labeling data

    label = []
    while True:
        try:
            next_ch = s[idx+1]
        except:
            label.append('<non_split>')
            break
        if next_ch == ' ':
            label.append('<split>')
            while s[idx+1] == ' ':
                idx += 1
            idx += 1
        else:
            label.append('<non_split>')
            idx += 1
    return label


filenames = glob('s/*.txt')

data = []
for p in tqdm(filenames):
    treg = re.compile('<\w*>|</\w*>')
    wreg = re.compile(' ')
    with open(p, 'r', encoding='utf-8') as f:
        a = f.readlines()
        a = [treg.sub('', str(t)).strip() for t in a]
        a = [t.lower() for t in a if t]
        labels = [get_label(t) for t in a]
        a = [wreg.sub('', str(t)) for t in a]
        for i in range(len(a)):
            data.append(([[char for char in a[i]], labels[i]]))


# split data to train_data(76%), val_data(4%), test_data(20%)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=77)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=7)


punctuations = [char for char in '~!@#$%&*()-=+{};:\'"<>,./?']
numbers = [char for char in '0123456789']
counter = Counter()

# counting number of appearance
for i in tqdm(range(len(train_data))):
    counter += Counter(train_data[i][0])

chars = []

# choose characters which will be in token_vocab
for key, val in counter.items():
    if val > 20:
        if ord(key) in range(0xac00, 0xd7a4) or ord(key) in range(0x0041, 0x007b):
            chars.append(key)

chars += punctuations
chars += numbers
chars.sort()


# defining vocab
label_vocab = Vocab(['<non_split>', '<split>'], unknown_token=None)
token_vocab = Vocab(chars)


# Saving Dataset, Vocab
print('Saving ...')

with open('dataset/train_data.pkl', mode='wb') as f:
    pickle.dump(train_data, f)

with open('dataset/val_data.pkl', 'wb') as f:
    pickle.dump(val_data, f)

with open('dataset/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

with open('dataset/label_vocab.pkl', 'wb') as f:
    pickle.dump(label_vocab, f)

with open('dataset/token_vocab.pkl', 'wb') as f:
    pickle.dump(token_vocab, f)
