import torch
import torch.nn as nn


class CoAttent(nn.Module):

    def __init__(self, num_vocab, embbed_size=512):

        self.embedding = nn.Embedding(num_vocab, embbed_size)
        # self.unigram = nn.Conv1d()