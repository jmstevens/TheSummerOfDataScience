import tensorflow as tf
import pandas as pd
import numpy as np
import os
import codecs
import collections

class DataProvider:
    data_dir = '../data/'
    def __init__(self, batch_size, sequence_length, artist):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.lyrics_raw = pd.read_csv(self.data_dir + 'lyrics.csv', low_memory=False)
        self.lyrics_artist = self.lyrics_raw[self.lyrics_raw['artist'] == artist]['lyrics'].dropna().values
        np.savetxt(os.path.join(self.data_dir, artist + '.txt'),self.lyrics_artist, fmt='%s')

        with codecs.open(os.path.join(self.data_dir, artist + '.txt'), "r", encoding="utf-8") as file:
            data = file.read()

        count_pairs = sorted(collections.Counter(data).items(), key=lambda x: -x[1])
        self.pointer = 0
        self.chars, _ = zip(*count_pairs)
        self.vocabularly_size = len(self.chars)
        self.vocabularly = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.array(list(map(self.vocabularly.get, data)))
        self.batches_size = int(self.tensor.size / (self.batch_size * self.sequence_length))

        if self.batches_size == 0:
            assert False, "Unable to generate batches. Reduce size or sequence length"

        self.tensor = self.tensor[:self.batches_size * self.batch_size * self.sequence_length]
        inputs = self.tensor
        targets = np.copy(self.tensor)

        targets[:-1] = inputs[1:]
        targets[-1] = inputs[0]

        self.input_batches = np.split(inputs.reshape(self.batch_size, -1), self.batches_size, 1)
        self.target_batches = np.split(targets.reshape(self.batch_size, -1), self.batches_size, 1)

        print("Tensor size: " + str(self.tensor.size))
        print("Batch size: " + str(self.batch_size))
        print("Sequence length: " + str(self.sequence_length))
        print("Batches size: " + str(self.batches_size))
        print("")

    def next_batch(self):
        inputs = self.input_batches[self.pointer]
        targets = self.target_batches[self.pointer]
        self.pointer += 1
        return inputs, targets

    def reset_batch_pointer(self):
        self.pointer = 0
