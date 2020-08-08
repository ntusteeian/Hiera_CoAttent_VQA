import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from prepro.build_dataset import VQADataset

class CoAttentModel(nn.Module):

    def __init__(self, num_vocab, embbed_size=512):

        super(CoAttent, self).__init__()

        resnet18 = model.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2]) # remove the last layer

        # Hierarchy question encoding
        self.embedding = nn.Embedding(num_vocab, embbed_size)
        self.unigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=1, stride=1, padding=0)
        self.bigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=2, stride=1, padding=1)
        self.trigram = nn.Conv1d(embbed_size, embbed_size, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d((3, 1))
        self.lstm = nn.LSTM(input_size=embbed_size, hidden_size=embbed_size, batch_first=True, dropout=0.5)

        # attention
        self.W_b = nn.Linear(in_features= embbed_size, out_features=embbed_size)
        # activation
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()

    def forward(self, img, qst, qst_length):

        word = self.embedding(qst)                              # [batch_size, qst_len, embbed_size=512]
        word = self.tanh(word).permute(0, 2, 1)                 # [batch_size, embbed_size=512, qst_len]

        unigram = self.unigram(word).unsqueeze(2)               # [batch_size, embbed_size=512, qst_len]
        bigram = self.bigram(word).narrow(2, 1, word.shape[-1]).unsqueeze(2)
        trigram = self.trigram(word).unsqueeze(2)

        phrase = torch.cat((unigram, bigram, trigram), dim=2)   # [batch_size, embbed_size=512, 3, qst_len]
        phrase = self.tanh(self.maxpool(phrase).squeeze())      # [batch_size, embbed_size=512, qst_len]
        phrase = self.dropout(phrase).permute(0, 2, 1)          # [batch_size, qst_len, embbed_size=512]

        qst_length = qst_length.tolist()
        phrase_packed = pack_padded_sequence(phrase, qst_length, batch_first=True, enforce_sorted=False) # [batch_sum_seq_len X embedding_dim]
        qst_packed, _ = self.lstm(phrase_packed)
        qst_embedding, _ = pad_packed_sequence(qst_packed, batch_first=True)  # [batch_size, max_qst_len=9, embbed_dim=512]

        with torch.no_grad():
            img_feat = self.resnet18(img)
        img_feat = img_feat.view(img_feat.size(0), img_feat.size(1), -1)

        return phrase

    def co_attention(self, img_feat, qst_feat):

        img_w = self.W_b(img_feat.permute(0, 2, 1))                # [batch_size, 512, 196]
        C = torch.bmm(qst_feat, img_w)

from torch.utils.data import DataLoader
from torchvision import transforms



if __name__ == '__main__':

    torch.manual_seed(0)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1

    ])
    vqa = VQADataset('../data', 'train.npy', transform=transform)
    num_vocab = vqa.qst_vocab.vocab_size
    train_loader = DataLoader(dataset=vqa,batch_size=5,shuffle=True,num_workers=8)
    test_model = CoAttent(num_vocab=num_vocab)

    for idx, sample in enumerate(train_loader):
        output = test_model(sample['image'], sample['question'], sample['length'])
