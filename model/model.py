import torch
import torch.nn as nn


class CoAttent(nn.Module):

    def __init__(self, num_vocab, embbed_size=512):

        super(CoAttent, self).__init__()
        self.num_vocab = num_vocab

        self.embedding = nn.Embedding(num_vocab, embbed_size)
        self.unigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=1, stride=1, padding=0)
        self.bigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=2, stride=1, padding=1)
        self.trigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=3, stride=1, padding=1)

    def forward(self, question):

        # embedding = self.embedding(question)
        a = self.bigram(question).narrow(2, 1, self.num_vocab)  # get 1: num_vocab+1 vector

        return a
if __name__ == '__main__':

    torch.manual_seed(0)

    test_model = CoAttent(num_vocab=10)
    input = torch.ones(1,10,512)  # 10 words, 512 vector
    input = input.permute(0,2,1) # 1 x 512 x 10
    output = test_model(input)
    print(output.size())