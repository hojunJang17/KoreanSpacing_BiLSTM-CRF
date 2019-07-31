import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.data import Corpus
from model.net import BiLSTM_CRF
from model.utils import batchify


def evaluate(model, data_loader, device):
    if model.training:
        model.eval()

    model.eval()
    avg_loss = 0
    for steps, mb in tqdm(enumerate(data_loader), desc='eval_steps', total=len(data_loader)):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)
        with torch.no_grad():
            mb_loss = model.loss(x_mb, y_mb)
        avg_loss += mb_loss.item()
    else:
        avg_loss /= (steps + 1)

    return avg_loss


with open('experiment/config.json') as f:
    params = json.loads(f.read())


# loading vocabs (token_vocab, label_vocab)
token_vocab_path = params['filepath'].get('token_vocab')
label_vocab_path = params['filepath'].get('label_vocab')
with open(token_vocab_path, 'rb') as f:
    token_vocab = pickle.load(f)
with open(label_vocab_path, 'rb') as f:
    label_vocab = pickle.load(f)


# defining model
hidden_size = params['model'].get('lstm_hidden_size')

model = BiLSTM_CRF(label_vocab, token_vocab, hidden_size)

# training parameters
epochs = params['training'].get('epochs')
batch_size = params['training'].get('batch_size')
learning_rate = params['training'].get('learning_rate')
global_step = params['training'].get('global_step')


# creating dataset, dataloader
train_path = params['filepath'].get('train')
val_path = params['filepath'].get('val')
train_data = Corpus(train_path, token_vocab.to_indices, label_vocab.to_indices)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16,
                          drop_last=True, collate_fn=batchify)
val_data = Corpus(val_path, token_vocab.to_indices, label_vocab.to_indices)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16,
                        drop_last=True, collate_fn=batchify)

# training
opt = optim.Adam(params=model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(opt, patience=5)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in tqdm(range(epochs), desc='epochs'):
    tr_loss = 0
    model.train()
    for step, mb in tqdm(enumerate(train_loader), desc='train_steps', total=len(train_loader)):

        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)

        opt.zero_grad()

        mb_loss = model.loss(x_mb, y_mb)
        mb_loss.backward()
        opt.step()

        tr_loss += mb_loss.item()

        if (epoch * len(train_loader) + step) % global_step == 0:
            val_loss = evaluate(model, val_loader, device)
            model.train()

    else:
        tr_loss /= (step+1)

    val_loss = evaluate(model, val_loader, device)
    scheduler.step(val_loss)

    tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch+1, tr_loss, val_loss))

# saving model
ckpt = {'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict()}
save_path = params['filepath'].get('ckpt')
torch.save(ckpt, save_path)