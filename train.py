import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import log
import numpy as np
import time
from sys import argv
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import pickle
import datetime
from dataset import produce_datasets, compute_binary_median_frequency_balancing, CollatePad
from nets import RNNMultiBinaryClassificationNet


precomputed_positive_class_weights = torch.tensor([9.41494656, 98.42056075, 17.82831858,
                                                   329.71502591, 19.27895155, 115.05090909])


def load_data(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds_train, ds_test = produce_datasets('jigsaw-toxic-comment-classification-challenge/train.csv',
                                         'jigsaw-toxic-comment-classification-challenge/vocab.txt',
                                         max_size=None,
                                         split_ratio=args.test_ratio)
    args.vocab_size = len(ds_train.vocab)
    args.padding_idx = ds_train.vocab.label_to_index['<PAD>']
    collate_fn = CollatePad(pad_value=args.padding_idx)
    train_loader = DataLoader(ds_train, shuffle=True, batch_size=args.bs, collate_fn=collate_fn)
    test_loader = DataLoader(ds_test, shuffle=False, batch_size=args.bs_val, collate_fn=collate_fn)
    if precomputed_positive_class_weights is not None:
        args.positive_class_weights = precomputed_positive_class_weights
    else:
        args.positive_class_weights = compute_binary_median_frequency_balancing(ds_train)
    args.positive_class_weights = args.positive_class_weights.to(args.device)
    return train_loader, test_loader


def load_model(args):
    print("Loading model...", end='')
    rnn_model = RNNMultiBinaryClassificationNet(num_embeddings=args.vocab_size, embedding_dim=args.embedding_dim,
                                                 num_tasks=6, padding_idx=args.padding_idx, p_dropout=args.dropout,
                                                 rnn_layers=len(args.rnn_sizes), hidden_size=args.rnn_sizes[0],
                                                 additional_fc_layer=args.additional_fc, dev=args.device)
    rnn_model.to(args.device)
    adam = Adam(rnn_model.parameters(), lr=args.lr, weight_decay=args.reg)
    print('done. (number of learnable parameters: {})'.format(sum(p.numel() for p in rnn_model.parameters()
                                                                  if p.requires_grad)))
    return rnn_model, adam


def train_evaluate(loaders: tuple, model: torch.nn.Module, optimizer: torch.optim.Optimizer, args):

    train_loader, test_loader = loaders
    classification_thresholds = torch.tensor([0.8, 0.98, 0.9, 0.99, 0.91, 0.97]).to(args.device)

    experiment_folder = "./runs/{}".format(datetime.datetime.now())
    writer = SummaryWriter(experiment_folder)
    with open(experiment_folder + "/config.pkl", "wb") as f:
        pickle.dump(args, f)

    print("Starting training...")
    np.set_printoptions(4)
    best_test_loss = np.inf
    best_f1_score = 0
    for epoch in range(args.epochs):

        model.train()
        t_start = time.time()
        for j, ((tokens, targets), (input_lengths, _)) in enumerate(train_loader):

            optimizer.zero_grad()

            # move to GPU
            tokens = tokens.to(args.device)
            targets = targets.float().to(args.device)

            # forward
            output = model(tokens, input_lengths)
            loss = -1/len(targets) * (args.positive_class_weights*targets*log(output) + (1-targets)*log(1-output)).sum()
            loss_value = loss.item()

            # backward
            loss.backward()

            # update step
            optimizer.step()

            if (j + 1) % args.log_every == 0:
                writer.add_scalar("Loss/train", loss_value, global_step=epoch * len(train_loader) + j)
                print("\rEpoch %3d/%3d, loss: %2.6f, "
                      "batch: %3d/%3d, pad length: %4d" % (epoch + 1, args.epochs, loss_value, j + 1,
                                                           len(train_loader), max(input_lengths)), end='')

        # evaluation
        epoch_duration = time.time() - t_start
        print("\nEpoch completed in {:3.2f}s. "
              "Evaluating (thresholds: {})...\r".format(epoch_duration,
                                                        classification_thresholds.cpu().numpy()), end='')

        model.eval()
        test_loss = 0
        total_correct = torch.zeros((6,))
        total_true_positives = torch.zeros((6,))
        total_false_positives = torch.zeros((6,))
        total_real_positives = torch.zeros((6,))
        for j, ((tokens, targets), (input_lengths, _)) in enumerate(test_loader):

            # move to GPU
            tokens = tokens.to(args.device)
            targets = targets.float().to(args.device)
            targets_b = targets.bool()

            # forward
            output = model(tokens, input_lengths)

            # compute stats
            y_pred = (output > classification_thresholds)
            total_correct += (y_pred == targets_b).sum(dim=0).cpu().float()
            total_true_positives += ((y_pred == 1) & (targets_b == 1)).sum(dim=0).cpu().float()
            total_false_positives += ((y_pred == 1) & (targets_b == 0)).sum(dim=0).cpu().float()
            total_real_positives += (targets_b == 1).sum(dim=0).cpu().float()
            loss = -1/len(targets) * (args.positive_class_weights*targets*log(output) + (1-targets)*log(1-output)).sum()
            test_loss += loss.item()

        test_loss /= len(test_loader)
        accuracies = (total_correct.numpy() / len(test_loader.dataset))
        precisions = (total_true_positives / (total_true_positives + total_false_positives)).numpy()
        recalls = (total_true_positives / total_real_positives).numpy()
        f1_scores = 2 * precisions * recalls / (precisions + recalls)
        avg_f1_score = np.average(f1_scores)

        writer.add_scalar("Loss/test", test_loss, global_step=(epoch + 1) * len(train_loader))
        writer.add_scalar("F1/test/average", avg_f1_score, global_step=(epoch + 1))
        writer.add_embedding(mat=model.embedding.weight.data, metadata=train_loader.dataset.vocab.label_to_index.keys(),
                             global_step=(epoch + 1) * len(train_loader))
        print("\nEvaluation completed.\nTest loss:\t{:2.6f}\naccuracies:\t{}\nprecisions:\t{}\n"
              "recalls:\t{}\nF1 scores:\t{}".format(test_loss, accuracies, precisions, recalls, f1_scores))

        # adapt classification thresholds
        precision_recall_distance = torch.tensor(precisions - recalls).to(args.device)
        classification_thresholds -= (1 - classification_thresholds) * precision_recall_distance

        # save
        if test_loss < best_test_loss or avg_f1_score > best_f1_score:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
            if avg_f1_score > best_f1_score:
                best_f1_score = best_f1_score
            print("Saving...")
            torch.save(model, "{}/epoch={}_loss={:.4f}_f1={:.4f}.pt".format(experiment_folder,
                                                                            epoch, test_loss, avg_f1_score))
        if epoch % args.save_every == 0:
            print("Saving...")
            torch.save(model, "{}/restore.pt".format(experiment_folder))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--test-ratio", default=0.2, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--lr-decay", default=1, type=float)  # TODO
    parser.add_argument("--decay-every", default=10000, type=int)  # TODO
    parser.add_argument("--reg", default=0.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--bs-val", default=256, type=int)
    parser.add_argument("--embedding-dim", default=64, type=int)
    parser.add_argument("--rnn-sizes", default=[128, 128], type=int, nargs="+")
    parser.add_argument("--additional-fc", metavar="ADDITIONAL_FC_SIZE", default=None, type=int)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--reload", metavar="MODEL", default=None, type=str)  # TODO
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv[1:])

    assert all(s == arguments.rnn_sizes[0] for s in arguments.rnn_sizes[1:]), "Only equally-sized stacked LSTMs allowed"
    arguments.device = torch.device("cuda") if torch.cuda.is_available() and not arguments.cpu else torch.device("cpu")

    data_loaders = load_data(arguments)
    model_, optimizer_ = load_model(arguments)
    train_evaluate(data_loaders, model_, optimizer_, arguments)
