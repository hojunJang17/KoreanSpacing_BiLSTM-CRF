import torch.nn as nn
from model.ops import BiLSTM, CRF


class BiLSTM_CRF(nn.Module):
    """
    Bidirectional LSTM - CRF
    Args:
        label_vocab (dict): mapping between labels and indices
        token_vocab (dict): mapping between tokens and indices
        hidden_size (int): hidden layer size
        embedding_dim (int): dimension of embedding layer
    """
    def __init__(self, label_vocab, token_vocab, hidden_size, embedding_dim=3):
        super().__init__()
        self.emb = nn.Embedding(len(token_vocab), embedding_dim, padding_idx=token_vocab.to_indices(token_vocab.padding_token))
        self.bilstm = BiLSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, len(label_vocab))
        self.crf = CRF(len(label_vocab), bos_tag_id=label_vocab.to_indices(label_vocab.bos_token),
                       eos_tag_id=label_vocab.to_indices(label_vocab.eos_token),
                       pad_tag_id=label_vocab.to_indices(label_vocab.padding_token),
                       batch_first=True)

    def forward(self, x):
        masking = x.ne(self.emb.padding_idx).float()
        x = self.emb(x)
        hiddens = self.bilstm(x)
        emissions = self.fc(hiddens)
        score, path = self.crf.decode(emissions, mask=masking)
        return score, path

    def loss(self, x, y):
        masking = x.ne(self.emb.padding_idx).float()
        x = self.emb(x)
        hiddens = self.bilstm(x)
        emission = self.fc(hiddens)
        nll = self.crf(emission, y, mask=masking)
        return nll
