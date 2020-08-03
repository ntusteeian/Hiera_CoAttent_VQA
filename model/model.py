import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class CoAttent(nn.Module):

    def __init__(self, num_vocab, embbed_size=512):

        super(CoAttent, self).__init__()

        self.embedding = nn.Embedding(num_vocab, embbed_size)
        self.unigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=1, stride=1, padding=0)
        self.bigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=2, stride=1, padding=1)
        self.trigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d((3, 1))
        self.tanh = nn.Tanh()

    def forward(self, question):

        word = self.embedding(question)                         # [batch_size, qst_len, embbed_dim=512]
        word = self.tanh(word).permute(0, 2, 1)                 # [batch_size, embbed_dim=512, qst_len]

        unigram = self.unigram(word).unsqueeze(2)               # [batch_size, embbed_dim=512, qst_len]
        bigram = self.bigram(word).narrow(2, 1, word.shape[-1]).unsqueeze(2)
        trigram = self.trigram(word).unsqueeze(2)

        phrase = torch.cat((unigram, bigram, trigram), dim=2)   # [batch_size, embbed_dim=512, 3, qst_len]
        phrase = self.tanh(self.maxpool(phrase).squeeze())      # [batch_size, embbed_dim=512, qst_len]
        phrase = self.dropout(phrase)
        print(phrase.size())
        return phrase

if __name__ == '__main__':

    torch.manual_seed(0)

    test_model = CoAttent(num_vocab=100)
    input = torch.ones(10, 5).type(torch.LongTensor)  # batch x question_len
    output = test_model(input)


    # maxpool = nn.MaxPool2d((3,1))
    # input = torch.rand(1, 2, 3, 8).unsqueeze(2) #batch channel height width
    # print(input.shape)
    # test = torch.unsqueeze(input, -1)
    # print(test.shape)
    # output = maxpool(input)
    # print(output.shape)
    # print(output)
