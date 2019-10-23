import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import log
import numpy as np
import time
from dataset import produce_datasets, compute_binary_median_frequency_balancing, CollatePad
from nets import LSTMMultiBinaryClassificationNet
from tensorboardX import SummaryWriter

num_epochs = 100
lr = 3e-4
bs = 32
bs_val = 128
log_every = 1
test_ratio = 0.2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
precomputed_positive_class_weights = torch.tensor([9.41494656, 98.42056075, 17.82831858,
                                                   329.71502591, 19.27895155, 115.05090909])
classification_thresholds = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).to(device)


def load_data():
    ds_train, ds_test = produce_datasets('jigsaw-toxic-comment-classification-challenge/train.csv',
                                         'jigsaw-toxic-comment-classification-challenge/vocab.txt',
                                         split_ratio=test_ratio)
    padding_idx = ds_train.vocab.label_to_index['<PAD>']
    collate_fn = CollatePad(pad_value=padding_idx)
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=bs, collate_fn=collate_fn)
    test_loader = DataLoader(ds_test, shuffle=False, batch_size=bs_val, collate_fn=collate_fn)
    if precomputed_positive_class_weights is not None:
        positive_class_weights = precomputed_positive_class_weights
    else:
        positive_class_weights = compute_binary_median_frequency_balancing(ds_train)
    positive_class_weights = positive_class_weights.to(device)
    return (train_loader, test_loader), positive_class_weights


def load_model():
    print("Loading model...", end='')
    lstm_model = LSTMMultiBinaryClassificationNet(num_embeddings=len(ds_train.vocab), embedding_dim=128,
                                                  num_classes=6, padding_idx=padding_idx, lstm_layers=2,
                                                  hidden_size=512, p_dropout=0.5, additional_fc_layer=None, dev=device)
    net.to(device)
    adam = Adam(model.parameters(), lr=lr)
    print('done.')
    return lstm_model, adam


def train_evaluate(loaders: tuple, model: torch.nn.Module, optimizer: torch.optim.Optimizer):

    train_loader, test_loader = loaders

    print("Starting training...")
    writer = SummaryWriter()
    np.set_printoptions(4)
    best_test_loss = np.inf
    best_f1_score = 0
    for epoch in range(num_epochs):

        model.train()
        t_start = time.time()
        for j, ((tokens, targets), (input_lengths, _)) in enumerate(train_loader):

            optimizer.zero_grad()

            # move to GPU
            tokens = tokens.to(device)
            targets = targets.float().to(device)

            # forward
            output = model(tokens, input_lengths)
            loss = -1 / len(targets) * (
                        positive_class_weights * targets * log(output) + (1 - targets) * log(1 - output)).sum()
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
        test_loss = 0
        total_correct = torch.zeros((6,))
        total_true_positives = torch.zeros((6,))
        total_false_positives = torch.zeros((6,))
        total_real_positives = torch.zeros((6,))
        for j, ((tokens, targets), (input_lengths, _)) in enumerate(test_loader):
            # move to GPU
            tokens = tokens.to(device)
            targets = targets.float().to(device)
            targets_b = targets.byte()

            # forward
            output = model(tokens, input_lengths)

            # compute stats
            y_pred = (output > classification_thresholds)
            total_correct += (y_pred == targets_b).sum(dim=0).cpu().float()
            total_true_positives += ((y_pred == 1) & (targets_b == 1)).sum(dim=0).cpu().float()
            total_false_positives += ((y_pred == 1) & (targets_b == 0)).sum(dim=0).cpu().float()
            total_real_positives += (targets_b == 1).sum(dim=0).cpu().float()
            loss = -1 / len(targets) * (
                        positive_class_weights * targets * log(output) + (1 - targets) * log(1 - output)).sum()
            test_loss += loss.item()

        test_loss /= len(test_loader)
        accuracies = (total_correct.numpy() / len(ds_test))
        precisions = (total_true_positives / (total_true_positives + total_false_positives)).numpy()
        recalls = (total_true_positives / total_real_positives).numpy()
        f1_scores = 2 * precisions * recalls / (precisions + recalls)

        writer.add_scalar("Loss/val", test_loss, global_step=(epoch + 1) * len(train_loader))
        writer.add_embedding(mat=model.embedding.weight.data, metadata=ds_train.vocab.label_to_index.keys(),
                             global_step=(epoch + 1) * len(train_loader))
        print("\nEvaluation completed.\nTest loss:\t{:2.6f}\naccuracies:\t{}\nprecisions:\t{}\n"
              "recalls:\t{}\nF1 scores:\t{}".format(test_loss, accuracies, precisions, recalls, f1_scores))

        # save
        avg_f1_score = np.average(f1_scores)
        if test_loss < best_test_loss or avg_f1_score > best_f1_score:
            best_test_loss = test_loss
            torch.save(model, "loss={:.4f}_f1={:.4f}.pt".format(test_loss, avg_f1_score))


if __name__ == '__main__':

    data_loaders, class_weights = load_data()
    model_, optimizer_ = load_model()
    train_evaluate(data_loaders, model_, optimizer_)
