import os
import json
import re
import argparse



def build_qst_vocab(input_dir, output_dir):

    dataset = os.listdir(input_dir)
    regex = re.compile(r"(\W+)")
    q_vocab = []
    for file in dataset:

        path = os.path.join(input_dir, file)
        with open(path, 'r') as f:
            q_data = json.load(f)
        question = q_data['questions']
        for idx, quest in enumerate(question):

            split = regex.split(quest['question'].lower())
            tmp = [w.strip() for w in split if len(w.strip()) > 0]
            q_vocab.extend(tmp)

    q_vocab = list(set(q_vocab))
    q_vocab.sort()
    q_vocab.insert(0, '<unk>')

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    with open(output_dir + '/question_vocabs.txt', 'w') as f:
        f.writelines([v+'\n' for v in q_vocab])

    print(f"total question word:{len(q_vocab)}")


def main(args):

    build_qst_vocab(args.input_qst_dir, args.output_dir)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_qst_dir', default='/HDD-1_data/dataset/VQA-v2/Questions',
                        help='input question directory from VQA v2.0 dataset')
    parser.add_argument('--output_dir', default='../data',
                        help='the directory of saving the build vocab')

    args = parser.parse_args()
    main(args)