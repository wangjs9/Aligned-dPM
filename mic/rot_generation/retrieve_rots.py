import pandas as pd
from transformers import Trainer
import argparse
from datasets import load_dataset, load_metric
import pandas as pd
from collections import OrderedDict, Counter
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union
import nltk, os, csv, json, random
import numpy as np
import torch
import json
from sentence_transformers import SentenceTransformer

SBERT = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def cosine_similarity(M, N):
    # normalize
    normalize = lambda X: (X.T / np.linalg.norm(X, axis=1))
    M_norm = normalize(M)
    N_norm = normalize(N)
    return np.matmul(M_norm.T, N_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to input file')
    parser.add_argument('--output', type=str, default='results', help='path to directory for outputting results')
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    df = pd.read_csv(args.input)
    train = df[df['split'] == 'train'].copy()
    test = df[df['split'] == 'test'].copy()

    print('embedding...')
    embeddings = {
        which: {
            col: SBERT.encode(df[df['split'] == which][col].values)
            for col in ['Q', 'A']
        }
        for which in ['train', 'test']
    }

    print('computing similarities...')
    similarities = {
        col: cosine_similarity(embeddings['train'][col], embeddings['test'][col])
        for col in ['Q', 'A']
    }

    sim = (similarities['Q'] + similarities['A']) / 2
    best_train_rot = train.iloc[np.argmax(sim, axis=0)]['rot'].values
    random_train_rot = train.sample_rot(n=len(test), replace=True, random_state=args.seed)['rot'].values

    print('writing to file...')
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    out_df = test[['Q', 'A', 'QA']].copy()
    out_df['rot'] = test['rot'].values
    out_df['rot_generated'] = random_train_rot
    out_df.to_csv(os.path.join(args.output, 'test_retrieval_random.csv'))

    out_df['rot_generated'] = best_train_rot
    out_df.to_csv(os.path.join(args.output, 'test_retrieval_SBERT.csv'))


if __name__ == '__main__':
    main()