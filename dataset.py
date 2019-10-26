from torch.utils.data import Dataset
import csv
from nltk.tokenize import word_tokenize
import numpy as np
import torch
import os


def compute_binary_median_frequency_balancing(dataset):

    print("Computing class weights...")
    num_tasks = len(dataset[0][1])
    frequencies = np.zeros((num_tasks, 2))
    for sample, targets in dataset:
        for j, target in enumerate(targets.int()):
            frequencies[j, target] += 1

    print(frequencies)
    positive_class_weights = frequencies[:, 0] / frequencies[:, 1]
    print(positive_class_weights)
    return torch.tensor(positive_class_weights, dtype=torch.float)


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
            padded_tensor = torch.mul(torch.ones((batch_size, *single_tensor_shape), dtype=torch.long), self.pad_value)
            for batch_index, length in enumerate(variable_lengths[:, group_index]):
                padded_tensor[batch_index, :length] = tensors[batch_index]
            padded_batch.append(padded_tensor)
            lengths.append(torch.tensor(variable_lengths[:, group_index], dtype=torch.long))
        return padded_batch, lengths


def compute_vocab(csv_file, dst_file, tokenizer=word_tokenize, max_size=65536):

    # load csv
    with open(csv_file, 'r') as csvf:
        csv_reader = csv.reader(csvf, delimiter=',')
        rows = list(csv_reader)

    # compute vocab
    all_words = [token for row in rows for token in map(lambda x: x.lower(), tokenizer(row[1]))]
    occurrences = dict()
    for word in all_words:
        occurrences[word] = occurrences.get(word, 0) + 1
    if max_size is None:
        max_size = len(occurrences)

    vocab = set(map(lambda x: x[0], sorted(occurrences.items(), key=lambda x: x[1], reverse=True)[:max_size - 2]))
    vocab.add('<PAD>')
    vocab.add('<UNK>')
    vocab = LabelIndexMap.from_list_of_labels(list(vocab), required_mappings={'<PAD>': 0, '<UNK>': 1})

    # save vocab
    with open(dst_file, 'w') as f:
        for label, index in vocab.label_to_index.items():
            f.write("{} {}\n".format(label, index))


class LabelIndexMap(object):
    """
    Class used to remap N labels from cluttered, arbitrary values to non-negative, contiguous values in range [0, N-1].
    """
    def __init__(self, dict_of_values):
        self.label_to_index = dict_of_values
        self.index_to_label = {index: label for label, index in dict_of_values.items()}

    @staticmethod
    def from_list_of_labels(all_labels, sort_key=None, required_mappings=None):
        assert (sort_key is None or required_mappings is None), "use only one among sort_key, predefined_positions"
        individual_labels = list(set(all_labels))
        if sort_key is not None:
            individual_labels = sorted(all_labels, key=sort_key)
        if required_mappings is not None:
            # set items to the required positions
            for element, required_position in required_mappings.items():
                if required_position >= len(individual_labels):
                    raise ValueError("Required position is out of range")
                if required_mappings.get(individual_labels[required_position], None) == required_position \
                        and element != individual_labels[required_position]:
                    raise ValueError("Incompatible required mapping: "
                                     "label '{}' and '{}' both required "
                                     "in position {}".format(element, individual_labels[required_position],
                                                             required_position))

                # swap
                current_position = individual_labels.index(element)
                individual_labels[current_position] = individual_labels[required_position]
                individual_labels[required_position] = element

        return LabelIndexMap({label: i for i, label in enumerate(individual_labels)})

    def __getitem__(self, item):
        return self.label_to_index[item]

    def __len__(self):
        return len(self.label_to_index)

    def save(self, filename):
        with open(filename, "w") as f:
            for item in self.label_to_index.items():
                f.write("{}\t{}\n".format(*item))

    @staticmethod
    def load(filename):
        v = dict()
        with open(filename, "r") as f:
            for row in f:
                label, index = row.split()
                v[label] = int(index)
        return LabelIndexMap(dict_of_values=v)


class ToxicCommentDataset(Dataset):

    def __init__(self, rows: list, vocab: LabelIndexMap, max_sequence_length=None):
        # init
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length

        if max_sequence_length is not None:
            assert isinstance(max_sequence_length, int)

        self.samples = []
        for i, row in enumerate(rows):
            print("Preparing dataset ({:6d}/{:6d})".format(i+1, len(rows)), end='\r')
            text, classes = row[1], [float(x) for x in row[2:8]]
            tokens = [self.vocab.label_to_index.get(token.lower(), self.vocab.label_to_index['<UNK>'])
                      for token in word_tokenize(text)][:max_sequence_length]
            self.samples.append((tokens, classes))

    def __getitem__(self, index):
        tokens, classes = self.samples[index]
        return torch.tensor(tokens), torch.tensor(classes)

    def __len__(self):
        return len(self.samples)


def produce_datasets(csv_file, max_dataset_size=None, max_sequence_length=None, vocab_size=65536, split_ratio=0.2):

    # load csv
    print("Loading CSV file '{}'...".format(csv_file))
    with open(csv_file, 'r') as csvf:
        csv_reader = csv.reader(csvf, delimiter=',')
        rows = list(csv_reader)[1:]
    if max_dataset_size is not None:
        assert 0 < max_size < len(rows), "Invalid max_size"
        rows = rows[:max_dataset_size]

    # load vocab
    vocab_filename = "vocab_{}.txt".format(vocab_size)
    vocab_path = os.path.join(os.path.dirname(csv_file), vocab_filename)
    if not os.path.exists(vocab_path):
        print("File '{}' not found. Computing vocab from training CSV file...".format(vocab_filename))
        compute_vocab(csv_file, dst_file=vocab_path, max_size=vocab_size)
        print("Vocab saved at '{}'.".format(vocab_path))
    print("Loading vocab file '{}'...".format(vocab_filename))
    vocab = LabelIndexMap.load(vocab_path)

    # produce datasets
    train_size = int((1 - split_ratio) * len(rows))
    train_set, test_set = ToxicCommentDataset(rows[:train_size], vocab, max_sequence_length), \
                          ToxicCommentDataset(rows[train_size:], vocab, None)
    return train_set, test_set


if __name__ == '__main__':
    ds_train, ds_val = produce_datasets('jigsaw-toxic-comment-classification-challenge/train.csv')
    print(len(ds_train))
    print(len(ds_train.vocab))
    print(ds_train[34])
