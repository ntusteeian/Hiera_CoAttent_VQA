import os
import re
import json
import argparse

from collections import defaultdict
import numpy as np


def preprocess(img_dir, qst_dir, anno_dir, vocab_file):

    """
        preprocess vqa data into {'qst_id': qst_id, 'qst_tokens':qst_tokens, 'img_path':img_path, 'ans': valid_ans}
        and save them into train.npy, val.npy, test.npy and train_val.npy seperately
    """
    qst_path = {subtype[20:-19]: os.path.join(qst_dir, subtype) for subtype in os.listdir(qst_dir)}
    anno_path = {subtype[10:-21]: os.path.join(anno_dir, subtype) for subtype in os.listdir(anno_dir)}

    train_qst = json.load(open(qst_path['train'], 'r'))
    val_qst = json.load(open(qst_path['val'], 'r'))
    test_qst = json.load(open(qst_path['test'], 'r'))
    train_anno = json.load(open(anno_path['train'], 'r'))
    val_anno = json.load(open(anno_path['val'], 'r'))

    dataset = defaultdict(list)
    img_temp = 'COCO_{}_{:0>12d}.jpg'
    for i in range(len(train_qst['questions'])):

        qst_id = train_qst['questions'][i]['question_id']
        img_id = train_qst['questions'][i]['image_id']
        qst_tokens = tokenizer(train_qst['questions'][i]['question'])
        img_name = img_temp.format(train_qst['data_subtype'], img_id)
        img_path = os.path.join(img_dir, train_qst['data_subtype'], img_name)

        anno_ans = train_anno['annotations'][i]['answers']
        _, valid_ans = match_top_ans(vocab_file, anno_ans)
        dataset['train'].append({'qst_id': qst_id, 'qst_tokens':qst_tokens, 'img_path':img_path, 'ans': valid_ans})

    for i in range(len(val_qst['questions'])):

        qst_id = val_qst['questions'][i]['question_id']
        img_id = val_qst['questions'][i]['image_id']
        qst_tokens = tokenizer(val_qst['questions'][i]['question'])
        img_name = img_temp.format(val_qst['data_subtype'], img_id)
        img_path = os.path.join(img_dir, val_qst['data_subtype'], img_name)

        anno_ans = val_anno['annotations'][i]['answers']
        _, valid_ans = match_top_ans(vocab_file, anno_ans)
        dataset['val'].append({'qst_id': qst_id, 'qst_tokens':qst_tokens, 'img_path':img_path, 'ans': valid_ans})

    for i in range(len(test_qst['questions'])):

        qst_id = test_qst['questions'][i]['question_id']
        img_id = test_qst['questions'][i]['image_id']
        qst_tokens = tokenizer(test_qst['questions'][i]['question'])
        img_name = img_temp.format(test_qst['data_subtype'], img_id)
        img_path = os.path.join(img_dir, test_qst['data_subtype'], img_name)
        
        dataset['test'].append({'qst_id': qst_id, 'qst_tokens':qst_tokens, 'img_path':img_path})

    dataset['train_val'] = dataset['train'] + dataset['val']

    return dataset

def tokenizer(sentence):

    regex = re.compile(r'(\W+)')
    tokens = regex.split(sentence.lower())
    tokens = [w.strip() for w in tokens if len(w.strip()) > 0]
    return tokens

def match_top_ans(vocab_file, anno_ans):

    if not bool(match_top_ans.__dict__):
        with open(vocab_file, 'r') as f:
            match_top_ans.top_ans = {line.strip() for line in f}

    anno_ans = {ans['answer'] for ans in anno_ans}
    valid_ans = match_top_ans.top_ans & anno_ans

    if len(valid_ans) == 0:
        valid_ans = ['<unk>']

    return anno_ans, valid_ans


def main(args):

    dataset = preprocess(args.img_dir, args.qst_dir, args.anno_dir, args.anno_vocab)

    for key, value in dataset.items():
        np.save(os.path.join(args.output_dir, f'{key}.npy'), np.array(value))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/HDD-1_data/dataset/VQA_v2.0/Images/')
    parser.add_argument('--qst_dir', default='/HDD-1_data/dataset/VQA_v2.0/Questions/')
    parser.add_argument('--anno_dir', default='/HDD-1_data/dataset/VQA_v2.0/Annotations/')
    parser.add_argument('--output_dir', default='../data')
    parser.add_argument('--anno_vocab', default='../data/annotation_vocabs.txt')

    args = parser.parse_args()
    main(args)