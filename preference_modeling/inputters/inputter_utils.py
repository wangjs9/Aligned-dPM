# coding=utf-8
import gzip
import json
import os
import math
import random
import pickle
from functools import partial
from torch.utils.data import DataLoader, Sampler


def _norm(s):
    return ' '.join(s.strip().split())


class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """

    def __init__(self, lens, bucket_size, batch_size, droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i + self._bucket_size], key=lambda i: self._lens[i], reverse=True) for i in
                   range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i + self._batch_size] for bucket in buckets for i in
                   range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s // self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s / self._batch_size) for s in bucket_sizes)


class BucketingDataLoader(object):
    def __init__(self, toker, feature_dataset, batch_size, bucket=100, shuffle=True, **kwargs):
        assert "inputter_name" in kwargs, "inputter name should be provided"
        inputter_name = kwargs["inputter_name"]
        assert inputter_name in ["off", "mic", "esc", "single"], "undefined inputter name"
        if inputter_name == "off":
            assert "fine_task" in kwargs, "offensive fine task should be provided"
            fine_task = kwargs["fine_task"]
            with open(f'dataset/off/{fine_task}_train.pkl', 'rb') as f:
                self.data = pickle.load(f)
        elif inputter_name == "mic":
            with open(f'dataset/mic/mic_data_train.pkl', 'rb') as f:
                self.data = pickle.load(f)
        else:
            assert "fold_num" in kwargs, "fold number should be provided when inputter name is esc"
            fold_num = kwargs["fold_num"]
            valid_ids = json.load(open(f'dataset/esc/dev_index.json', 'r'))[fold_num]
            with open(f'dataset/esc/{inputter_name}_data.pkl', 'rb') as f:
                self.data = [line for line in pickle.load(f) if line.dialog_id not in valid_ids]

        self.toker = toker
        self.feature_dataset = feature_dataset
        self.batch_size = batch_size
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle

    def __iter__(self):
        trunc_chunk = []
        lens = []
        for feat in self.data:
            trunc_chunk.append(feat)
            lens.append(feat.input_length)

        dataset = self.feature_dataset(trunc_chunk)
        sampler = BucketSampler(lens, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0,  # can be multi-worker
                            collate_fn=partial(self.feature_dataset.collate, toker=self.toker))
        yield from loader

    def __len__(self):
        return len(self.data)


class DistributedBucketingDataLoader(BucketingDataLoader):
    """ distributed version """

    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.num_replica = num_replica
        self.data = self.data[self.rank::self.num_replica]
