import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Vocabulary:
    def __init__(self, sequences):
        # initiate the index to token dict
        self.itos = {}
        # initiate the token to index dict
        self.stoi = {}
        self._build_vocabulary(sequences)

    def _build_vocabulary(self, sequences):
        max_seq_len = 0
        idx = 1  # index from which we want our dict to start.
        # create vocab
        for sequence in sequences:
            if len(sequence) > max_seq_len:
                max_seq_len = len(sequence)
            unique_tokens_dict = OrderedDict((aa, idx) for aa in sequence)
            unique_tokens = list(unique_tokens_dict.keys())
            for token in unique_tokens:
                if token not in self.stoi.keys():
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1
        self.max_seq_len = max_seq_len
        # self.stop_index = len(self.itos) + 1
        # self.itos[self.stop_index] = "|"
        # self.stoi["|"] = self.stop_index

    def transform(self, sequences):  # for all sequences at once
        output = np.zeros((len(sequences), self.max_seq_len), dtype=np.int16)
        for i, sequence in enumerate(sequences):
            for j, s in enumerate(sequence):
                if s in self.stoi.keys():
                    output[i][j] = self.stoi[s]
                else:
                    raise KeyError(f"This vocabulary doesn't know this aminoacid: {s}")
            # else:
            #     output[i][j + 1] = self.stop_index
        return output

    def numericalize(self, sequence):  # for one sequence
        tokenized_sequence = self.tokenizer(sequence)  # 'ATY' -> ['A', 'T', 'Y']
        numericalized_sequence = []
        for token in tokenized_sequence:
            if token in self.stoi.keys():
                numericalized_sequence.append(
                    self.stoi[token]
                )  # ['A', 'T', 'Y'] -> [1, 13, 19] (ids of the corresponding AAs)
            else:
                raise KeyError(f"This vocabulary doesn't know this aminoacid: {token}")
        return numericalized_sequence

    def retrieve_sequence(self, seq_indices):
        decoded_pep = ""
        for idx in seq_indices:
            if idx == 0:
                break
            else:
                decoded_pep += self.itos[idx]
        return decoded_pep

    def __len__(self):
        return len(self.itos)


class SequenceDataset(Dataset):
    def __init__(
            self,
            file_name,
            sequence_column,
            vocabulary_path,
            label_column=None,
    ):
        super().__init__()
        self.data = pd.read_csv(file_name)
        self.num_seqs = self.data.shape[0]
        sequences = self.data[sequence_column].to_list()
        """
        vocabulary_path = Path(vocabulary_path)
        if vocabulary_path.exists():
            with open(vocabulary_path, "rb") as inp:
                self.vocabulary = pickle.load(inp)
        else:
            self.vocabulary = Vocabulary(sequences)
            with open(vocabulary_path, "wb") as out:
                pickle.dump(self.vocabulary, out)
        """
        vocabulary_path = Path(vocabulary_path)
        if vocabulary_path.exists() and vocabulary_path.stat().st_size > 0:
            print("vocab exists")
            try:
                with open(vocabulary_path, "rb") as inp:
                    self.vocabulary = pickle.load(inp)
                    print("pickle loaded")
            except EOFError:
                print("EOFError: The file is empty or incomplete")
            except Exception as e:
                print(f"Error during unpickling data: {e}")
        else:
            self.vocabulary = Vocabulary(sequences)
            try:
                with open(vocabulary_path, "wb") as out:
                    pickle.dump(self.vocabulary, out)
                    print("vocabulary pickled successfully")
            except Exception as e:
                print(f"Error during pickling data: {e}")

        print(len(self.vocabulary))

        self.sequences = torch.from_numpy(self.vocabulary.transform(sequences))
        self.sequences = self.sequences.type(torch.FloatTensor)

        self.stoi = self.vocabulary.stoi  # added by K 010223
        self.max_seq_len = self.vocabulary.max_seq_len
        # self.stop_index = self.vocabulary.stop_index
        self.vocabulary_size = (len(self.vocabulary.itos) + 1)  # we count padding here too that is why +1

        if label_column is not None:
            self.labels = torch.from_numpy(
                self.data[label_column].astype(float).to_numpy()
            )
            self.labels = self.labels.type(torch.FloatTensor)
            assert self.sequences.shape[0] == self.labels.shape[0]
        else:
            self.labels = None

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return self.sequences[idx]
        else:
            return self.sequences[idx], self.labels[idx]
