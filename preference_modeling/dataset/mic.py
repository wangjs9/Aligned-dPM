# coding=utf-8

import argparse
import os
import json
import pickle
import pandas as pd
from datasets import load_dataset, load_metric
from tqdm import tqdm
import multiprocessing as mp
from collections import defaultdict
from inputters import inputters

Inputter = inputters["mic"]()
from utils.building_utils import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--max_input_length', type=int, default=512, help='discard data longer than this')
parser.add_argument('--train_size', type=int, default=-1,
                    help='the number of train datapoints to use as an ablation (default is to use the full train set)')

args = parser.parse_args()


def process_raw():
    mic_datapath = "dataset/mic/MIC.csv"
    df = pd.read_csv(mic_datapath)
    for split in ['train', 'dev']:
        if (split == 'train') and (args.train_size > 0):
            df[df['split'] == split].sample_rot(n=args.train_size, random_state=args.seed).to_csv(
                f'dataset/mic/{split}.csv', index=False)
        else:
            df[df['split'] == split].to_csv(f'dataset/mic/{split}.csv', index=False)
    """ delete the original file to save space """
    os.remove(mic_datapath)


toker = build_model(only_toker=True)

SAVE_DIR = 'dataset/mic'
assert os.path.exists(SAVE_DIR)
if not os.path.exists("dataset/mic/train.csv"):
    process_raw()

kwargs = {
    'max_input_length': args.max_input_length,
}


def process_line(line):
    preference = Inputter.convert_data_to_inputs(data=line, toker=toker, **kwargs)
    features = Inputter.convert_inputs_to_features(inputs=preference, toker=toker, **kwargs)
    return features


dataset = load_dataset('csv', data_files={'train': "dataset/mic/train.csv",
                                          'dev': "dataset/mic/dev.csv"})
print('train size:', len(dataset['train']))

preference = []
with mp.Pool(processes=mp.cpu_count()) as pool:
    for features in pool.imap(process_line, tqdm(dataset['train'], total=len(dataset['train']))):
        preference.extend(features)

data_path = f"{SAVE_DIR}/mic_data_train.pkl"
with open(data_path, 'wb') as file:
    pickle.dump(preference, file)

print('training data saved to', data_path)

val_preference = []
with mp.Pool(processes=mp.cpu_count()) as pool:
    for features in pool.imap(process_line, tqdm(dataset['dev'], total=len(dataset['dev']))):
        val_preference.extend(features)

dev_data_path = f"{SAVE_DIR}/mic_data_dev.pkl"
with open(dev_data_path, 'wb') as file:
    pickle.dump(val_preference, file)

print('dev data saved to', dev_data_path)

""" delete files to save space """
os.remove("dataset/mic/train.csv")
os.remove("dataset/mic/dev.csv")
