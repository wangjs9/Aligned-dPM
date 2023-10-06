# coding=utf-8

import json
import tqdm
import torch
import pickle
from typing import List
from math import ceil
from functools import partial
from numpy.random import triangular

from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer

try:
    from preference_modeling.inputters.inputter_utils import _norm, BucketingDataLoader, DistributedBucketingDataLoader
except ModuleNotFoundError:
    from inputters.inputter_utils import _norm, BucketingDataLoader, DistributedBucketingDataLoader
import logging

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class Inputter(object):

    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features

        # train
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader

        # valid
        self.valid_dataloader = DynamicBatchingLoader

        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


# basic utils
class InputFeatures(object):
    def __init__(self, input_ids, token_type_ids, score):
        self.input_ids = input_ids
        self.input_length = len(input_ids)
        self.token_type_ids = token_type_ids
        self.score = score


def featurize(bos, eos, inputs, max_input_length=512):
    QA, rot, score = inputs["QA"], inputs["rot"], inputs["score"]
    rot = rot + [eos]
    rot_ids = rot[:max_input_length]
    QA_ids = [bos] + QA + [bos]
    input_ids = QA_ids + rot_ids
    input_ids = input_ids[-max_input_length:]
    token_type_ids = [0] * len(input_ids)
    token_type_ids[-len(rot_ids):] = [1] * len(rot_ids)

    assert len(input_ids) == len(token_type_ids)

    return InputFeatures(input_ids, token_type_ids, score)


# for training
def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    assert 'max_input_length' in kwargs, "max_input_length should be given"
    process = lambda x: toker.convert_tokens_to_ids(
        toker.tokenize(x, max_length=kwargs['max_input_length'], truncation=True)
    )
    certain_rate = {
        1: 0.025, 2: 0.15, 3: 0.5, 4: 0.85, 5: 0.95
        # 1: 0.025, 2: 0.15, 3: 0.5, 4: 0.85, 5: 0.95
    }
    QA = process(_norm(data["QA"]))
    rot = process(_norm(data["rot"]))
    severity = data["violation-severity"]
    certainty = certain_rate[int(data["rot-agree"])]
    if type(certainty) == float:
        score = certainty
    else:
        mode = severity / 6 * (certainty[1] - certainty[0]) + certainty[0]
        score = triangular(certainty[0], mode, certainty[1])

    inputs = [{
        "QA": QA,
        "rot": rot,
        "score": score,
    }]

    return inputs


def convert_inputs_to_features(inputs, toker: PreTrainedTokenizer, **kwargs):
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) != None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad != None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos != None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos != None, 'either eos_token_id or sep_token_id should be provided'
    features = []
    for i, ipt in enumerate(inputs):
        feat = featurize(bos, eos, ipt, max_input_length)
        features.append(feat)

    return features


# for training
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad != None, 'either pad_token_id or eos_token_id should be provided'
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
            assert bos != None, 'either bos_token_id or cls_token_id should be provided'
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id
            assert eos != None, 'either eos_token_id or sep_token_id should be provided'

        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features], batch_first=True,
                                 padding_value=pad)
        attention_mask = input_ids == pad
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids, dtype=torch.long) for f in features],
                                      batch_first=True, padding_value=pad)

        scores = pad_sequence([torch.tensor([f.score, 1 - f.score], dtype=torch.float) for f in features],
                              batch_first=True)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "scores": scores,
        }

        return res


# for validation
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """

    def __init__(self, toker, batch_size, **kwargs):
        with open(f'dataset/mic/mic_data_dev.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.trunc_chunk = []
        self.lens = []
        for feat in self.data:
            self.trunc_chunk.append(feat)
            self.lens.append(feat.input_length)

        self.toker = toker
        self.bs = batch_size
        self.num_examples = len(self.trunc_chunk)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for i_epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            features = []
            for feature in self.trunc_chunk:
                features.append(feature)
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []

        except StopIteration:
            pass

    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)


# for inference
def convert_infer_to_features(inputs, toker: PreTrainedTokenizer, **kwargs):
    if len(inputs) == 0:
        return {}

    assert kwargs.get('max_input_length', None) != None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad != None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos != None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos != None, 'either eos_token_id or sep_token_id should be provided'

    input_ids = [ipt + [eos] for ipt in inputs["inp_seq"]]
    input_ids = [ipt[-max_input_length:] for ipt in input_ids]

    decoder_input_ids = inputs["out_seq"]
    bos_tensor = torch.ones(decoder_input_ids.size(0), 1, dtype=decoder_input_ids.dtype) * bos
    decoder_input_ids = torch.cat((bos_tensor, decoder_input_ids), -1)

    features = {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
    }
    return features


# for inference
def prepare_infer_batch(features, pad=0):
    input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features], batch_first=True,
                             padding_value=pad)
    attention_mask = input_ids == pad
    token_type_ids = pad_sequence([torch.tensor(f.token_type_ids, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=pad)

    res = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    res['batch_size'] = res['input_ids'].size(0)

    return res


def get_infer_batch(corpus_df, toker, **kwargs):
    assert 'max_input_length' in kwargs, "max_input_length should be given"
    process = lambda x: toker.convert_tokens_to_ids(
        toker.tokenize(x, max_length=kwargs['max_input_length'], truncation=True)
    )

    if "no_bar_info" in kwargs:
        bar = enumerate(corpus_df)
    else:
        bar = tqdm.tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="preference score computing")

    for sample_id, line in bar:
        rot_samples = line.get("rot_samples", [])
        QA = process(_norm(line["QA"]))
        inputs = [{
            "QA": QA,
            "rot": process(_norm(line["rot"])),
            "score": None,
        }]
        for rot in rot_samples:
            inputs.append({
                "QA": QA,
                "rot": process(_norm(rot)),
                "score": None,
            })

        features = convert_inputs_to_features(inputs, toker, **kwargs)

        yield prepare_infer_batch(features, toker.pad_token_id), sample_id
