import numpy as np
import os

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


class VQADataset(Dataset):

    def __init__(self, input_dir, input_file, max_qst_len = 30, transform = None):

        self.input_data = np.load(os.path.join(input_dir, input_file), allow_pickle=True)
        self.qst_vocab = Vocab(input_dir+'/question_vocabs.txt')
        self.ans_vocab = Vocab(input_dir+'/annotation_vocabs.txt')
        self.max_qst_len = max_qst_len
        self.labeled = True if not "test" in input_file else False
        self.transform = transform

    def __getitem__(self, idx):

        path = self.input_data[idx]['img_path']
        img = Image.open(path).convert('RGB')
        qst_id = self.input_data[idx]['qst_id']
        qst_tokens = self.input_data[idx]['qst_tokens']
        qst2idx = np.array([self.qst_vocab.word2idx('<pad>')] * self.max_qst_len)
        qst2idx[:len(qst_tokens)] = [self.qst_vocab.word2idx(token) for token in qst_tokens]
        # sample = {'image': img, 'question': qst2idx, 'question_id': qst_id}
        sample = {'question': qst2idx, 'question_id': qst_id, 'length': len(qst_tokens)}
        sample = {'image': img, 'question': qst2idx, 'question_id': qst_id, 'length': len(qst_tokens)}

        if self.labeled:
            ans2idx = [self.ans_vocab.word2idx(ans) for ans in self.input_data[idx]['ans']]
            ans2idx = np.random.choice(ans2idx)
            sample['answer'] = ans2idx

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def __len__(self):

        return len(self.input_data)

def data_loader(input_dir, batch_size, max_qst_len, num_worker):

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1

    ])

    vqa_dataset = {
        'train': VQADataset(
            input_dir=input_dir,
            input_file='train.npy',
            max_qst_len=max_qst_len,
            transform=transform),
        'val': VQADataset(
            input_dir=input_dir,
            input_file='val.npy',
            max_qst_len=max_qst_len,
            transform=transform)
    }

    dataloader = {
        key: DataLoader(
            dataset=vqa_dataset[key],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_worker)
        for key in ['train', 'val']
    }

    return dataloader

class Vocab:

    def __init__(self, vocab_file):

        self.vocab = self.load_vocab(vocab_file)
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab_file):

        with open(vocab_file) as f:
            vocab = [v.strip() for v in f]

        return vocab

    def word2idx(self, vocab):

        if vocab in self.vocab2idx:
            return self.vocab2idx[vocab]
        else:
            return self.vocab2idx['<unk>']

    def idx2word(self, idx):

        return self.vocab[idx]
