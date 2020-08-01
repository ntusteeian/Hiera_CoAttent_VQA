import torch
import torch.nn as nn


class CoAttent(nn.Module):

    def __init__(self, num_vocab, embbed_size=512):

        self.embedding = nn.Embedding(num_vocab, embbed_size)
        self.unigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=1, stride=1, padding=0)
        self.bigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=2, stride=1, padding=1)
        self.trigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=2, stride=1, padding=2)

    def forward(self, input):

        pass

torch.manual_seed(0)
conv1 = nn.Conv1d(in_channels=5, out_channels=2, padding=0, kernel_size=2, dilation=1)
input = torch.ones(1,3,5)
print(input)
# batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
input = input.permute(0,2,1)
out = conv1(input)
print(conv1.weight)
print('sum0:',torch.sum(conv1.weight[0]))
print('sum1:',torch.sum(conv1.weight[1]))

print(conv1.bias)
print(out.size())
print(out)
