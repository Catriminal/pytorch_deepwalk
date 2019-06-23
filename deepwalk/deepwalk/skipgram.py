import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        huffman_node_sum = 2 * emb_size - 1
        self.center_embeddings = nn.Embedding(huffman_node_sum, emb_dimension, sparse=True)
        self.window_embeddings = nn.Embedding(huffman_node_sum, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.center_embeddings.weight.data.uniform_(-initrange, initrange)
        self.window_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_center, pos_window, neg_center, neg_window):
        losses = []
        pos_emb_center = self.center_embeddings(Variable(torch.LongTensor(pos_center)))
        pos_emb_window = self.window_embeddings(Variable(torch.LongTensor(pos_window)))
        pos_score = torch.mul(pos_emb_center, pos_emb_window)
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)
        losses.append(sum(pos_score))
        neg_emb_center = self.center_embeddings(Variable(torch.LongTensor(neg_center)))
        neg_emb_window = self.window_embeddings(Variable(torch.LongTensor(neg_window)))
        neg_score = torch.mul(neg_emb_center, neg_emb_window)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)

    def save_embedding(self, id2word, file_name):
        embedding = self.center_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding="UTF-8")
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, word in id2word.items():
            emb = embedding[wid]
            emb = ' '.join(map(lambda x: str(x), emb))
            fout.write('%s %s\n' % (word, emb))



