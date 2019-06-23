import numpy
from collections import deque
from deepwalk.huffman import HuffmanTree
numpy.random.seed(12345)


class InputData:
    def __init__(self, input_file, min_count):
        self.input_file = input_file
        self.sentence_sum_length = 0
        self.sentence_count = 0
        self.word_count = 0
        self.word2id = dict()
        self.id2word = dict()
        self.word_frequency = dict()
        self.word_pair_catch = deque()

        self.get_words(min_count)
        tree = HuffmanTree(self.word_frequency)
        self.huffman_positive, self.huffman_negative = tree.divide_pos_and_neg()

    def get_words(self, min_count):

        word_frequency = dict()
        input_file = open(self.input_file)
        for line in input_file:
            self.sentence_count += 1
            words = line.strip().split(' ')
            self.sentence_sum_length += len(words)
            for word in words:
                try:
                    word_frequency[word] += 1
                except:
                    word_frequency[word] = 1

        wid = 0
        for word, count in word_frequency.items():
            if count < min_count:
                self.sentence_sum_length -= count
                continue

            self.word2id[word] = wid
            self.id2word[wid] = word
            self.word_frequency[wid] = count
            wid += 1

        self.word_count = len(self.word2id)
        input_file.close()

    def get_batch_pairs(self, batch_size, window_size):
        input_file = open(self.input_file)
        while len(self.word_pair_catch) < batch_size:
            sentence = input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue

            for i, center_wid in enumerate(word_ids):
                window_wids = word_ids[max(i - window_size, 0): i + window_size]
                for j, window_wid in enumerate(window_wids):
                    assert center_wid < self.word_count
                    assert window_wid < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((center_wid, window_wid))

        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())

        input_file.close()
        return batch_pairs

    def get_pairs_from_huffman(self, word_pairs):
        pos_word_pairs = []
        neg_word_pairs = []
        for index in range(len(word_pairs)):
            pair = word_pairs[index]
            pos_word_pairs += zip([pair[0]] * len(self.huffman_positive[pair[1]]),
                                  self.huffman_positive[pair[1]])
            neg_word_pairs += zip([pair[0]] * len(self.huffman_negative[pair[1]]),
                                  self.huffman_negative[pair[1]])

        return pos_word_pairs, neg_word_pairs

    def evaluate_pair_count(self, window_size):
        return self.sentence_sum_length * (2 * window_size - 1) - (self.sentence_count - 1) * (1 + window_size) * window_size
