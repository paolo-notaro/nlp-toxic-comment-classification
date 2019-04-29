import torch
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from math import inf
import numpy as np
import time
from dataset import produce_datasets
from nets import LSTMClassificationNet
from tensorboardX import SummaryWriter

num_epochs = 100
lr = 3e-4
bs = 32
log_every = 1
val_ratio = 0.2
positive_threshold = 0.5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ratio_positive_examples = torch.tensor([0.0960, 0.0101, 0.0531, 0.0030, 0.0493, 0.0086]).to(device)
dataset_file_path = '/home/paulstpr/Downloads/WhatsApp Chat with Sara Pontelli 💙.txt'


class CollatePad(object):

    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, batch):
        """
        Collates groups of Tensors of variable lengths into one padded Tensor (for each group).
        :param batch: list of tuple of tensors (e.g. list of 32 tuples, where the first elements are inputs and the
        seconds are targets). Individual Tensors must be of shape (T, N1, N2, N3, ...) where T is the variable dimension
        and N1, N2, N3, ... are any number of additional dimensions of fixed size. Inside the final Tensor, elements
        will be sorted in descending order of length. IMPORTANT: if more than one group of Tensor has variable length,
        it is assumed that the order according to length is consistent across groups, i.e. the variable lengths inside
        the different groups of Tensor are equal along the same index in the group. Example: group 1 represents
        tokenized sentences [["Hi", "Paolo", speaking"], ["Hi", "how", "are", "you", "today"]], group 2 POS tags
        [[EXCL, NOM, VRB], [EXCL, ADV, BE, SUBJ, ADV]], lengths are consistent across groups thus the program can sort
         according to the length of item of any attribute.
        :return: tuple of:
            * list of Tensors containing the padded batch, one for each group;
            * list of Tensors containing the variable lengths, one for each group.
        """

        batch = sorted(batch, reverse=True, key=lambda elem: len(elem[0]))
        variable_lengths = np.array([[tensor.shape[0] for tensor in tensors] for tensors in batch])
        batch_size, num_tensors = variable_lengths.shape
        max_lengths = np.max(variable_lengths, axis=0)

        padded_batch = []
        lengths = []
        for group_index, max_length in enumerate(max_lengths):
            tensors = [tensors[group_index] for tensors in batch]
            single_tensor_shape = (max_length, *batch[0][group_index].shape[1:])
            padded_tensor = self.pad_value * torch.ones((batch_size, *single_tensor_shape), dtype=torch.long)
            for batch_index, length in enumerate(variable_lengths[:, group_index]):
                padded_tensor[batch_index, :length] = tensors[batch_index]
            padded_batch.append(padded_tensor)
            lengths.append(torch.tensor(variable_lengths[:, group_index], dtype=torch.long))
        return padded_batch, lengths


if __name__ == '__main__':

    ds_train, ds_val = produce_datasets('jigsaw-toxic-comment-classification-challenge/train.csv',
                                        'jigsaw-toxic-comment-classification-challenge/vocab.txt', val_ratio=val_ratio)

    padding_idx = ds_train.vocab.label_to_index['<PAD>']
    collate_fn = CollatePad(padding_idx)
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=bs, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, shuffle=False, batch_size=64, collate_fn=collate_fn)

    print("Loading model...", end='')
    model = LSTMClassificationNet(num_embeddings=len(ds_train.vocab), embedding_dim=64, num_classes=6,
                                  padding_idx=padding_idx, lstm_layers=1, hidden_size=512, p_dropout=0.5,
                                  additional_fc_layer=64, dev=device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    print('done.')

    print("Starting training...")
    writer = SummaryWriter()
    np.set_printoptions(4)
    best_val_loss = inf
    best_f1_score = 0
    for epoch in range(num_epochs):

        model.train()
        t_start = time.time()
        for j, ((tokens, targets), (input_lengths, _)) in enumerate(train_loader):

            optimizer.zero_grad()

            # move to GPU
            tokens = tokens.to(device)
            targets = targets.float().to(device)
            class_weights = torch.abs(targets - ratio_positive_examples)
            class_weights = class_weights/class_weights.sum(0)*tokens.shape[0]

            # forward
            output = model(tokens, input_lengths)
            loss = binary_cross_entropy(input=output, target=targets, weight=class_weights, reduction='mean')
            loss_value = loss.item()

            # backward
            loss.backward()

            # update step
            optimizer.step()

            if (j + 1) % log_every == 0:
                writer.add_scalar("Loss/train", loss_value, global_step=epoch * len(train_loader) + j)
                print("\rEpoch %3d/%3d, loss: %2.6f, "
                      "batch: %3d/%3d, pad length: %4d" % (epoch + 1, num_epochs, loss_value, j + 1,
                                                           len(train_loader), max(input_lengths)), end='')

        # evaluation
        epoch_duration = time.time() - t_start
        print("\nEpoch completed in {:3.2f}s. Evaluating...\r".format(epoch_duration), end='')
        model.eval()
        val_loss = 0
        total_correct = torch.zeros((6, ))
        total_true_positives = torch.zeros((6,))
        total_false_positives = torch.zeros((6,))
        total_real_positives = torch.zeros((6,))
        for j, ((tokens, targets), (input_lengths, _)) in enumerate(val_loader):

            # move to GPU
            tokens = tokens.to(device)
            targets = targets.float().to(device)
            class_weights = torch.abs(targets - ratio_positive_examples)
            class_weights = class_weights / class_weights.sum(0) * tokens.shape[0]
            targets_b = targets.byte()

            # forward
            output = model(tokens, input_lengths)

            # compute stats
            y_pred = (output > positive_threshold)
            total_correct += (y_pred == targets_b).sum(dim=0).cpu().float()
            total_true_positives += ((y_pred == 1) & (targets_b == 1)).sum(dim=0).cpu().float()
            total_false_positives += ((y_pred == 1) & (targets_b == 0)).sum(dim=0).cpu().float()
            total_real_positives += (targets_b == 1).sum(dim=0).cpu().float()
            loss = binary_cross_entropy(input=output, target=targets, weight=class_weights, reduction='mean')
            val_loss += loss.item()

        val_loss /= len(val_loader)
        accuracies = (total_correct.numpy() / len(ds_val))
        precisions = (total_true_positives / (total_true_positives + total_false_positives)).numpy()
        recalls = (total_true_positives / total_real_positives).numpy()
        f1_scores = 2*precisions*recalls/(precisions + recalls)

        writer.add_scalar("Loss/val", val_loss, global_step=(epoch + 1) * len(train_loader))
        writer.add_embedding(mat=model.embedding.weight.data, metadata=ds_train.vocab.label_to_index.keys(),
                             global_step=(epoch + 1) * len(train_loader))
        print("\nEvaluation completed.\nValidation loss:\t{:2.6f}\naccuracies:\t{}\nprecisions:\t{}\n"
              "recalls:\t{}\nF1 scores:\t{}".format(val_loss, accuracies, precisions, recalls, f1_scores))

        # save
        avg_f1_score = np.average(f1_scores)
        if val_loss < best_val_loss or avg_f1_score > best_f1_score:
            best_val_loss = val_loss
            torch.save(model, "loss={:.4f}_f1={:.4f}.pt".format(val_loss, avg_f1_score))
