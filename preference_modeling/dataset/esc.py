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

Inputter = inputters["esc"]()
from utils.building_utils import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--max_input_length', type=int, default=160, help='discard data longer than this')

args = parser.parse_args()


def process_raw(num_fold=10):
    csv_file = "dataset/esc/MI Dataset.csv"
    reader = pd.read_csv(csv_file, sep=',', header=0, index_col=None, encoding="UTF-8")
    labels = defaultdict(int)
    anno_idx = ["ann1", "ann2", "judge 1 annotation", "judge 2 annotation"]
    dialog_ids = list(set(reader['dialog_id']))
    interval = len(dialog_ids) // num_fold

    fold_index = []
    for i_fold in range(num_fold):
        end_point = (i_fold + 1) * interval if i_fold != num_fold - 1 else len(dialog_ids)
        temp_ids = dialog_ids[i_fold * interval: end_point]
        fold_index.append(temp_ids)
    json.dump(fold_index, open("dataset/esc/dev_index.json", "w"))

    writter = open("dataset/esc/mi_data.txt", "w")
    conversation = defaultdict(list)
    for line in reader.iterrows():
        line = line[1]
        dialog_id, turn_idx = line["dialog_id"], line["turn"]
        author = line["author"]
        if author == "speaker":
            if len(conversation) > 0 and "responses" in conversation:
                writter.write(json.dumps(conversation))
                writter.write("\n")
            conversation = defaultdict(list)
            conversation["dialog_id"] = dialog_id
            conversation["speaker_turn"] = turn_idx
            conversation["context"].append(line["text"])
        elif author == "listener":
            annotations = [line[x] for x in anno_idx]
            for annotation in annotations:
                if annotation != "-":
                    labels[f"[{annotation}]"] += 1
            conversation["responses"].append([line["text"]] + annotations)

    writter.close()
    json.dump(labels, open("dataset/esc/mi_labels.json", "w"))


for file_name in ["mi_data.txt", "dev_index.json", "mi_labels.json"]:
    if not os.path.exists(f".dataset/esc/{file_name}"):
        process_raw()
        break

with open("dataset/esc/mi_labels.json", "r") as f:
    labels = json.load(f)
    labels = list(labels.keys())

toker = build_model(only_toker=True)

SAVE_DIR = 'dataset/esc'
assert os.path.exists(SAVE_DIR)

kwargs = {
    'max_input_length': args.max_input_length,
}


def process_line(line):
    data = json.loads(line)
    preference = Inputter.convert_data_to_inputs(data=data, toker=toker, **kwargs)
    features = Inputter.convert_inputs_to_features(inputs=preference, toker=toker, **kwargs)
    return features


with open("dataset/esc/mi_data.txt", "r") as f:
    reader = f.readlines()

preference = []
with mp.Pool(processes=mp.cpu_count()) as pool:
    for features in pool.imap(process_line, tqdm(reader, total=len(reader))):
        preference.extend(features)

data_path = f"{SAVE_DIR}/esc_data.pkl"
with open(data_path, 'wb') as file:
    pickle.dump(preference, file)
