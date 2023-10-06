# coding=utf-8

import json
import tqdm
import random
import torch
import pickle
from typing import List
from math import ceil
from functools import partial

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
    def __init__(self, dialog_id, input_ids, token_type_ids, label):
        self.dialog_id = dialog_id
        self.input_ids = input_ids
        self.input_length = len(input_ids)
        self.token_type_ids = token_type_ids
        self.label = label


def featurize(bos, eos, inputs, max_input_length=512):
    dialog_id, context, response = inputs["dialog_id"], inputs["context"], inputs["response"]
    label = inputs["label"]
    response = response + [eos]
    response_ids = response[:max_input_length]
    context_ids = [bos] + context + [bos]
    input_ids = context_ids + response_ids
    input_ids = input_ids[-max_input_length:]
    token_type_ids = [0] * len(input_ids)
    token_type_ids[-len(response_ids):] = [1] * len(response_ids)

    assert len(input_ids) == len(token_type_ids)

    return InputFeatures(dialog_id, input_ids, token_type_ids, label)


# for training
def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    assert 'max_input_length' in kwargs, "max_input_length should be given"
    process = lambda x: toker.convert_tokens_to_ids(
        toker.tokenize(x, max_length=kwargs['max_input_length'], truncation=True)
    )
    token_num = toker.vocab_size
    fine_to_coarse = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 13: 0, 14: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 2, -100: -100
    }

    context = _norm(' '.join(data["context"]))
    context = process(context)
    dialog_id = data["dialog_id"]
    responses = data['responses']
    inputs = []

    for reply_anno in responses:
        fine_ids = [process(f"[{anno}]")[0] - token_num for anno in reply_anno[1:] if anno != "-"]
        for label in fine_ids:
            inputs.append({
                "dialog_id": dialog_id,
                "context": context,
                "response": process(_norm(reply_anno[0])),
                "label": fine_to_coarse[label],
            })

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

        labels = torch.tensor([f.label for f in features], dtype=torch.long)

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

        return res


# for dev
class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """

    def __init__(self, toker, batch_size, **kwargs):
        assert "fold_num" in kwargs, "fold_num should be given"
        fold_num = kwargs["fold_num"]
        dev_index = json.load(open("dataset/esc/dev_index.json", 'r'))[fold_num]
        with open("dataset/esc/single_data.pkl", 'rb') as f:
            self.data = pickle.load(f)
        self.trunc_chunk = []
        self.lens = []
        for feat in self.data:
            if feat.dialog_id in dev_index:
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


def get_infer_batch(corpus, toker, **kwargs):
    assert "max_input_length" in kwargs, "max_input_length should be given"
    process = lambda x: toker.convert_tokens_to_ids(
        toker.tokenize(x, max_length=kwargs["max_input_length"], truncation=True)
    )

    if "no_bar_info" in kwargs:
        bar = enumerate(corpus)
    else:
        bar = tqdm.tqdm(enumerate(corpus), total=len(corpus), desc="preference score computing")

    for sample_id, line in bar:
        data = json.loads(line)
        context = _norm(' '.join(data["context"]))
        context = process(context)
        responses = data['responses']
        inputs = []
        for reply in responses:
            inputs.append({
                "dialog_id": sample_id,
                "context": context,
                "response": process(_norm(reply)),
                "label": None,
            })
        features = convert_inputs_to_features(inputs, toker, **kwargs)

        yield prepare_infer_batch(features, toker.pad_token_id), sample_id
