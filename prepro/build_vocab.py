import os
import json
import re
import argparse
from collections import defaultdict


def tokenize(sentence):

    regex = re.compile(r'(\W+)')
    tokens = regex.split(sentence.lower())
    tokens = [w.strip() for w in tokens if len(w.strip()) > 0]
    return tokens

def build_qst_vocab(input_dir, output_dir):

    qst_file = os.listdir(input_dir)
    q_vocab = []

    for file in qst_file:
        path = os.path.join(input_dir, file)
        with open(path, 'r') as f:
            q_data = json.load(f)
        question = q_data['questions']
        for idx, quest in enumerate(question):
            tokens = tokenize(quest['question'])
            q_vocab.extend(tokens)

    q_vocab = list(set(q_vocab))
    q_vocab.sort()
    q_vocab.insert(0, '<pad>')
    q_vocab.insert(1, '<unk>')

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(output_dir + '/question_vocabs.txt', 'w') as f:
        f.writelines([v+'\n' for v in q_vocab])

    print(f"total question word:{len(q_vocab)}")

def build_ans_vocab(input_dir, output_dir, num_ans):

    answers = defaultdict(lambda :0)
    dataset = os.listdir(input_dir)
    for file in dataset:
        path = os.path.join(input_dir, file)
        with open(path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        for label in annotations:
            for ans in label['answers']:
                vocab = ans['answer']
                answers[vocab] += 1

    top_ans = sorted(answers, key=answers.get, reverse= True) # sort by numbers
    top_ans = ['<unk>'] + top_ans[:num_ans-1]
    with open(output_dir + '/annotation_vocabs.txt', 'w') as f :
        f.writelines([ans+'\n' for ans in top_ans])

    print(f'The number of total words of answers: {len(answers)}')
    print('top answers and their counts:')

    for idx, ans in enumerate(top_ans):
        print(f'top {idx} answer:{ans:<20}, count: {answers[ans]}')


def main(args):

    build_qst_vocab(args.input_qst_dir, args.output_dir)
    build_ans_vocab(args.input_ans_dir, args.output_dir, args.top_ans)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--input_qst_dir', default='/HDD-1_data/dataset/VQA_v2.0/Questions',
                        help='input question directory from VQA v2.0 dataset')
    parser.add_argument('--input_ans_dir', default='/HDD-1_data/dataset/VQA_v2.0/Annotations',
                        help='input annotations directory from VQA v2.0 dataset')
    parser.add_argument('--output_dir', default='../data',
                        help='the directory of saving the build vocab')
    # optional
    parser.add_argument('--top_ans', default=1000, type=int, help='number of top answers for the final classifications.')

    args = parser.parse_args()
    main(args)