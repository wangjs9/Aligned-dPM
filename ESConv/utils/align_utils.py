# coding=utf-8

import os
import sys
import json
import pickle
import numpy as np
from typing import List
from functools import partial
import datetime

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer

from inputters import inputters
from inputters.inputter_utils import BucketSampler, _norm
from utils.eval_utils import eval_model_loss

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from metric.myMetrics import Metric


def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent


def candidate_sampling(model, toker, args, **infer_dataloader_kwargs):
    """
    This function is to generate candidates for each post in the training set.

    :param model: the base model --> Blender-Vanilla or Blender-Joint
    :param toker: the tokenizer
    :param args: hyper-parameters including candidate_num, checkpoint_dir, train_input_file, inputter, and device.
    :return: the directory of the generated candidates.
    """
    model.eval()
    inputter = inputters[args.inputter_name]()
    sample_dataloader = inputter.sample_dataloader(args.train_input_file, toker, **infer_dataloader_kwargs)

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

    candidate_num = args.candidate_num
    sampling_kwargs = {
        'max_length': 40,
        'min_length': 10,
        'do_sample': False,
        'temperature': 0.7,
        'top_k': 0,
        'top_p': 0.9,
        'num_beams': candidate_num,
        'num_beam_groups': candidate_num,
        'num_return_sequences': candidate_num,
        'length_penalty': 1.0,
        'repetition_penalty': 1.0,
        'diversity_penalty': 4.0,
        'no_repeat_ngram_size': 0,
        'encoder_no_repeat_ngram_size': 3,
        'pad_token_id': pad,
        'bos_token_id': bos,
        'eos_token_id': eos,
    }
    print(json.dumps(sampling_kwargs, indent=2, ensure_ascii=False))

    results = []
    decode = lambda x: _norm(toker.decode(x))
    for batch, posts, speakers, references, sample_ids in sample_dataloader:
        batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        batch.update(sampling_kwargs)
        encoded_info, generations = model.generate(**batch)
        generations = generations.tolist()
        generations = [cut_seq_to_eos(each, eos) for each in generations]

        for idx in range(len(sample_ids)):
            p = posts[idx]
            r = references[idx]
            s = speakers[idx]
            if candidate_num is not None:
                g = []
                for gg in generations[idx * candidate_num: (idx + 1) * candidate_num]:
                    g.append(gg)
            else:
                g = generations[idx]

            if isinstance(g[0], list):
                g = [decode(gg) for gg in g]
            else:
                g = [decode(g)]
            tmp_res_to_append = {
                "sample_id": sample_ids[idx],
                "context": p[-3:],
                "speaker": s[-3:],
                "responses": [r] + g
            }
            results.append(tmp_res_to_append)

    save_dir = args.candidate_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "candidates.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, sort_keys=False)

    with open(os.path.join(save_dir, "candidates.txt"), "w") as f:
        for line in results:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    return os.path.join(save_dir, "candidates.json")


def test_model(model, toker, args, loss_loader, infer_dataloader):
    model.eval()
    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

    generation_kwargs = {
        'max_length': 40,
        'min_length': 10,
        'do_sample': True,
        'temperature': 0.7,
        'top_k': 0,
        'top_p': 0.9,
        'num_beams': 1,
        'num_return_sequences': 1,
        'length_penalty': 1.0,
        'repetition_penalty': 1.0,
        'no_repeat_ngram_size': 0,
        'encoder_no_repeat_ngram_size': 3,
        'pad_token_id': pad,
        'bos_token_id': bos,
        'eos_token_id': eos,
    }
    print(json.dumps(generation_kwargs, indent=2, ensure_ascii=False))

    metric_results = {}
    infer_loss, _, infer_samples, pointwise_loss, pointwise_sample = eval_model_loss(
        model=model,
        eval_dataloader=loss_loader,
        infer=True,
        args=args,
    )
    assert len(pointwise_loss) == len(pointwise_sample)
    metric_results["perplexity"] = float(np.exp(infer_loss))
    ptr = 0
    metric = Metric(toker)

    results = []
    decode = lambda x: _norm(toker.decode(x))

    with torch.no_grad():
        for batch, contexts, references, sample_ids in infer_dataloader:
            batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            batch.update(generation_kwargs)
            encoded_info, generations = model.generate(**batch)

            generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]

            for idx in range(len(sample_ids)):
                c = contexts[idx]
                r = references[idx]
                g = generations[idx]
                ref, gen = [r], toker.decode(g) if not isinstance(g[0], list) else toker.decode(g[0])
                metric.forword(ref, gen, chinese=args.chinese)
                g = decode(g)
                tmp_res_to_append = {"sample_id": sample_ids[idx], "context": c, "response": r, "generation": g}

                ptr_loss = pointwise_loss[ptr]
                ptr_sample = pointwise_sample[ptr]
                turn_loss = ptr_loss / ptr_sample
                turn_ppl = np.exp(turn_loss)
                tmp_res_to_append["token_num"] = ptr_sample
                tmp_res_to_append["loss"] = turn_loss
                tmp_res_to_append["ppl"] = turn_ppl
                ptr += 1
                results.append(tmp_res_to_append)

    assert ptr == len(pointwise_loss)

    save_dir = os.path.join(args.checkpoint_dir, f"inference_results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "test_generations.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, sort_keys=False)

    with open(os.path.join(save_dir, "test_generations.txt"), "w") as f:
        for line in results:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    metric_result_list = {}
    closed_result = metric.close()
    metric_results.update(closed_result[0])
    metric_result_list.update(closed_result[1])

    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metric_results, f, indent=2, ensure_ascii=False, sort_keys=False)

    if metric_result_list != None:
        with open(os.path.join(save_dir, "test_metrics_list.json"), "w", encoding="utf-8") as f:
            json.dump(metric_result_list, f, indent=2, ensure_ascii=False, sort_keys=False)


def eval_model(model, toker, args, loss_loader, infer_dataloader):
    model.eval()

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

    generation_kwargs = {
        'max_length': 40,
        'min_length': 10,
        'do_sample': True,
        'temperature': 0.7,
        'top_k': 0,
        'top_p': 0.9,
        'num_beams': 1,
        'num_return_sequences': 1,
        'length_penalty': 1.0,
        'repetition_penalty': 1.0,
        'no_repeat_ngram_size': 0,
        'encoder_no_repeat_ngram_size': 3,
        'pad_token_id': pad,
        'bos_token_id': bos,
        'eos_token_id': eos,
    }
    infer_loss, infer_ppl, infer_samples, pointwise_loss, pointwise_sample = eval_model_loss(
        model=model,
        eval_dataloader=loss_loader,
        infer=True,
        args=args,
    )
    assert len(pointwise_loss) == len(pointwise_sample)
    metric_results = {"perplexity": float(np.exp(infer_loss))}
    ptr = 0
    metric = Metric(toker)

    results = []
    decode = lambda x: _norm(toker.decode(x))

    with torch.no_grad():
        for batch, contexts, references, sample_ids in infer_dataloader:
            batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            batch.update(generation_kwargs)
            encoded_info, generations = model.generate(**batch)

            generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]

            for idx in range(len(sample_ids)):
                c = contexts[idx]
                r = references[idx]
                g = generations[idx]
                ref, gen = [r], toker.decode(g) if not isinstance(g[0], list) else toker.decode(g[0])
                metric.forword(ref, gen, chinese=args.chinese)
                g = decode(g)
                tmp_res_to_append = {"sample_id": sample_ids[idx], "context": c, "response": r, "generation": g}

                ptr_loss = pointwise_loss[ptr]
                ptr_sample = pointwise_sample[ptr]
                turn_loss = ptr_loss / ptr_sample
                turn_ppl = np.exp(turn_loss)
                tmp_res_to_append["token_num"] = ptr_sample
                tmp_res_to_append["loss"] = turn_loss
                tmp_res_to_append["ppl"] = turn_ppl
                ptr += 1
                results.append(tmp_res_to_append)

    assert ptr == len(pointwise_loss)

    metric_result_list = {}
    closed_result = metric.close()
    metric_results.update(closed_result[0])
    metric_result_list.update(closed_result[1])

    return infer_loss, infer_ppl, metric_results, metric_result_list, results


class AlignmentFeature(object):
    def __init__(self, feature, candidates, max_length, inputter_is_strat=False):
        self.input_ids = feature.input_ids
        self.input_length = feature.input_length

        if inputter_is_strat:
            strat = torch.tensor([feature.decoder_input_ids[:2]] * candidates.size(0), dtype=candidates.dtype)
            self.candidates = torch.cat((strat, candidates[:, 1:]), dim=-1)[:, :max_length]
        else:
            self.candidates = candidates[:, :max_length]

        self.input_len = feature.input_len


class AlignmentDataLoader(object):
    def __init__(self, toker, data_dir, max_length=40, bucket=100, is_sorted=True, max_num=5, batch_size=1,
                 shuffle=True, preference_mark="d-PM", **kwargs):
        """ :returns format: context, golden_response, [[candidiate_i, score_i], ...] """
        self.max_length = max_length
        self.batch_size = batch_size
        self.toker = toker
        self.sorted = is_sorted
        self.bucket_size = bucket * self.batch_size
        self.maxnum = max_num
        self.shuffle = shuffle

        # load generated responses and corresponding scores
        input_score_file = os.path.join(data_dir, f"preference_score_{preference_mark}.npy")
        input_generation_file = os.path.join(data_dir, "candidates.txt")
        assert os.path.exists(input_score_file)
        scores = np.load(input_score_file, allow_pickle=True)
        with open(input_generation_file, "r", encoding='utf-8') as file:
            gens = [json.loads(line)["responses"] for line in file]
        golden = [gg[0] for gg in gens]

        # candidates are ranked according to best --> worst
        cands = [{r: s for r, s in zip(rlst[1:], slst[1:]) if s > 1e-7} for rlst, slst in zip(gens, scores)]
        cands = [[gg] + [x[0] for x in sorted(cand.items(), key=lambda x: x[1])] for gg, cand in zip(golden, cands)]
        # load processed data
        assert "inputter_name" in kwargs
        assert "config_name" in kwargs
        inputter_name = kwargs.pop("inputter_name")
        config_name = kwargs.pop("config_name")
        with open(f"DATA/{inputter_name}.{config_name}/data.pkl", "rb") as f:
            data = pickle.load(f)
        self.trunc_chunk, self.lens = self.process_data(data, cands, inputter_name == "strat")

    def process_data(self, data, cands, inputter_is_strat=False):
        trunc_chunk = []
        lens = []
        for feat, cand in zip(data, cands):
            cand = self.toker.batch_encode_plus(cand, return_tensors="pt", pad_to_max_length=False, padding=True)
            cand = cand["input_ids"]
            bos_tensor = torch.ones(cand.size(0), 1, dtype=cand.dtype) * self.toker.bos_token_id
            new_cand = torch.cat((bos_tensor, cand), -1)
            feat = AlignmentFeature(feat, new_cand, self.max_length, inputter_is_strat)
            trunc_chunk.append(feat)
            lens.append(feat.input_len)

        return trunc_chunk, lens

    def __len__(self):
        return len(self.trunc_chunk)

    def __iter__(self):
        dataset = AlignmentDataset(self.trunc_chunk)
        sampler = BucketSampler(self.lens, self.bucket_size, self.batch_size, droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4,  # can test multi-worker
                            collate_fn=partial(dataset.collate, toker=self.toker))
        yield from loader


class AlignmentDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[AlignmentFeature], toker: PreTrainedTokenizer):
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'

        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features], batch_first=True,
                                 padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                                      batch_first=True, padding_value=0.)
        cand_max_len = max([f.candidates.size(1) for f in features])
        # candidate_ids = [pad_to_max(f.candidates, pad, cand_max_len) for f in features]
        if len(features) > 1:
            cand_max_num = max([f.candidates.size(0) for f in features])
            cand_total_max = cand_max_len * cand_max_num
            candidate_ids = [pad_to_max(f.candidates, pad, cand_max_len).reshape(1, -1) for f in features]
            candidate_ids = [pad_to_max(cand, pad, cand_total_max).reshape(cand_max_num, -1) for cand in candidate_ids]
        else:
            candidate_ids = [pad_to_max(f.candidates, pad, cand_max_len) for f in features]
        ##### delete 419-423
        candidate_ids = torch.stack(candidate_ids)
        res = {"input_ids": input_ids, "attention_mask": attention_mask, "candidate_ids": candidate_ids}

        return res


def pad_to_max(X, pad_token_id, max_len=-1):
    if max_len < 0:
        max_len = max(x.size(0) for x in X)
    seq_num, seq_len = X.size()
    if seq_len == max_len:
        return X
    else:
        pad_tensor = torch.ones(seq_num, (max_len - seq_len), dtype=X[0].dtype) * pad_token_id
        result = torch.cat((X, pad_tensor), -1)
        return result


def RankingLoss(score, gold_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    if score.sum() == 0:
        return 0
    loss_func = torch.nn.MarginRankingLoss(0.0, reduction='sum')
    loss_mask = (score != 0).long()
    TotalLoss = loss_func(score, score, loss_mask) / loss_mask.sum()
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            # ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='sum')
            loss_mask = ((pos_score != 0) & (neg_score != 0)).long()
            loss = loss_func(pos_score, neg_score, loss_mask) / (loss_mask.sum() + 1e-8)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold response loss
    if gold_weight > 0:
        pos_score = gold_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        # ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(gold_margin, reduce='sum')
        loss_mask = (neg_score != 0).long()
        TotalLoss += gold_weight * loss_func(pos_score, neg_score, loss_mask) / loss_mask.sum()
    return TotalLoss
