import torch
import json
from torch.utils.data import DataLoader
from model.utils import batchify
import itertools
from model.data import Corpus
from model.net import BiLSTM_CRF
from sklearn.metrics import f1_score
from tqdm import tqdm
import pickle

def get_eval(model, data_loader, device):
    # Evaluation method : f1 score

    if model.training:
        model.eval()

    true_entities = []
    pred_entities = []

    for mb in tqdm(data_loader, desc='steps'):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)
        y_mb = y_mb.cpu()
        with torch.no_grad():
            _, yhat = model(x_mb)
            pred_entities.extend(list(itertools.chain.from_iterable(yhat)))
            true_entities.extend(y_mb.masked_select(y_mb.ne(0)).numpy().tolist())
    else:
        score = f1_score(true_entities, pred_entities, average='weighted')
    return score


with open('experiment/config.json') as f:
    params = json.loads(f.read())

# loading token & label vocab
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

# loading datasets
batch_size = params['training'].get('batch_size')
train_path = params['filepath'].get('train')
val_path = params['filepath'].get('val')
test_path = params['filepath'].get('test')
train_data = Corpus(train_path, token_vocab.to_indices, label_vocab.to_indices)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16,
                          drop_last=True, collate_fn=batchify)
val_data = Corpus(val_path, token_vocab.to_indices, label_vocab.to_indices)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16,
                        drop_last=True, collate_fn=batchify)
test_data = Corpus(test_path, token_vocab.to_indices, label_vocab.to_indices)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16,
                         drop_last=True, collate_fn=batchify)

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


# evaluating
print('Evaluating train set')
train_f1_score = get_eval(model, train_loader, device)

print('Evaluating validation set')
val_f1_score = get_eval(model, val_loader, device)

print('Evaluating test set')
test_f1_score = get_eval(model, test_loader, device)

print('Train Set f1_score: {:.2%}'.format(train_f1_score))
print('Validation Set f1_score: {:.2%}'.format(val_f1_score))
print('Test Set f1_score: {:.2%}'.format(test_f1_score))