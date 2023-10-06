#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
import torch
import logging
from torch import Tensor
import numpy as np
from collections import defaultdict
from sklearn import metrics

logger = logging.getLogger(__name__)


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g) - n):
                ngram = ' '.join(g[idx:idx + n + 1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v + 0.0) / total * (np.log(v + 0.0) - np.log(total))
        div_score[n] = (len(counter[n].values()) + 0.0) / total
    return etp_score, div_score


def eval_model(model, eval_dataloader, infer, args):
    logger.info('\ncompute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = 0
    tot_sample = 0
    pointwise_loss = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

            outputs = model(**batch)
            loss_sample = outputs["loss"]
            if torch.isnan(loss_sample).sum().cpu().long().numpy() > 0:
                print(loss_sample)
                exit()
            num_sample = batch["input_ids"].shape[0]
            tot_loss += loss_sample.sum().item()
            tot_sample += num_sample

            if infer:
                pointwise_loss.extend(loss_sample.sum(dim=-1).cpu().tolist())
    mean_loss = tot_loss / tot_sample
    return mean_loss, pointwise_loss
