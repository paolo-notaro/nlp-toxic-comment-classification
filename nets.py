import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
        for name, param in m._parameters.items():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param)
            elif "bias" in name:
                torch.nn.init.constant_(param, 0)


class RNNMultiBinaryClassificationNet(nn.Module):

    def __init__(self, num_tasks, num_embeddings, embedding_dim, hidden_size, rnn_layers,
                 p_dropout, padding_idx, dev, additional_fc_layer=None):

        super().__init__()
        self.num_classes = num_tasks
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.p_dropout = p_dropout
        self.device = dev

        # layers
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=rnn_layers,
                          batch_first=True)
        if additional_fc_layer is not None:
            assert isinstance(additional_fc_layer, int)
            self.fc = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(hidden_size, additional_fc_layer),
                                    nn.ReLU(), nn.Dropout(), nn.Linear(additional_fc_layer, num_tasks), nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Dropout(p_dropout), nn.Linear(hidden_size, num_tasks), nn.Sigmoid())

        self.hidden_state = None
        self.reset_state()
        self.apply(kaiming_init)

    def forward(self, batch, actual_lengths=None, reset_state=True):

        bs, pad_length = batch.shape
        if actual_lengths is None:
            actual_lengths = torch.tensor([pad_length]*bs, dtype=torch.long).to(self.device)

        if reset_state:
            self.reset_state(bs=bs)
        else:
            assert all(self.hidden_state[i].shape[1] == bs for i in range(2))

        x = self.embedding(batch)

        x = pack_padded_sequence(x, actual_lengths, batch_first=True)
        z, self.hidden_state = self.gru(x, self.hidden_state)
        z, _ = pad_packed_sequence(z, batch_first=True)
        last_z = z[range(len(actual_lengths)), actual_lengths - 1, :]

        z = self.fc(last_z)
        return z

    def reset_state(self, bs=1):
        self.hidden_state = torch.zeros((self.rnn_layers, bs, self.hidden_size)).to(self.device)
