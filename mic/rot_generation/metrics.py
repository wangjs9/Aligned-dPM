import os
import pandas as pd
import numpy as np
import bert_score
import sacrebleu
import json
from rouge_score import rouge_scorer
from glob import glob
import argparse
import re, nltk
from datasets import load_metric

ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)


def rouge_max(cands, refs):
    scores = {}

    preprocess = lambda x: "\n".join(nltk.sent_tokenize(x.strip()))
    for metric in ROUGE_SCORER.rouge_types:
        max_scores = np.array(
            [max([ROUGE_SCORER.score(preprocess(ref), preprocess(cand))[metric].fmeasure for ref in refs])
             for cand, refs in zip(cands, refs)])
        scores[metric] = np.mean(np.array(max_scores))
    return scores


def sacrebleu_max(cands, refs):
    sacrebleu = load_metric('sacrebleu')
    bleu = load_metric('bleu')
    # results = sacrebleu.compute(predictions=cands, references=refs)["score"]
    max_scores = np.array([
        sacrebleu.compute(predictions=[cand], references=[ref])["score"] for cand, ref in zip(cands, refs)
    ])
    return float(np.mean(np.array(max_scores))), 0


# df --> cands, refs
def get_cands_refs(df, prompt_col, cand_col, refs_col):
    cand_list = []
    refs_list = []
    for prompt in set(df[prompt_col].values):
        consider = df[df[prompt_col] == prompt].copy()
        refs = list(consider[refs_col].values)
        for cand in consider[cand_col].values:
            cand_list.append(cand)
            refs_list.append(refs)
    return cand_list, refs_list


def mean_length(sentences):
    return np.mean(np.array([len(nltk.tokenize.word_tokenize(sent)) for sent in sentences]))


def compute_metrics(df, prompt_col, cand_col, refs_col):
    cands, refs = get_cands_refs(df, prompt_col, cand_col, refs_col)
    # sacrebleu_max(cands, refs)
    scores = rouge_max(cands, refs)
    bert_p, bert_r, bert_f1 = bert_score.score(cands, refs, lang='en')
    scores['BERTScore_Precision'] = float(np.average(np.array(bert_p)))
    scores['BERTScore_Recall'] = float(np.average(np.array(bert_r)))
    scores['BERTScore'] = float(np.average(np.array(bert_f1)))
    scores['sacrebleu'], scores['bleu'] = sacrebleu_max(cands, refs)
    scores['mean_length'] = mean_length(df[cand_col].values)

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='glob regex for input models')
    parser.add_argument('--output', type=str, help='path to directory for outputting results')
    parser.add_argument('--target', default='rot', choices=['rot', 'target'])
    parser.add_argument('--prompt_col', type=str, default='QA')
    args = parser.parse_args()

    GEN_COL = 'rot_generated'

    data = {}
    i = 0
    for dirname in list(glob(args.input)):
        print(dirname)
        for fn in list(glob(f"{dirname}/test_generations*.csv")):
            tmp = fn.split('/')[-1]
            # result_path = fn[:-4] + f'_results.json'
            result_path = f"{dirname}/metric_{tmp[5:-4]}.json"
            if os.path.exists(result_path):
                print('skipping', fn)
                continue
            print('\t', fn)

            # remove brackets after splitting on <eos> and [attr] to gather only the RoT
            process = lambda txt: re.sub('[\s]*\[[\w/]+\][\s]*', '', re.sub(
                '[\s]*<[\w/]+>[\s]*', '', re.sub('^([\s]*<[\w/]+>[\s]*)+', '', txt).split('<eos>')[0].split('</s>')[0]
            ).strip())

            generations = pd.read_csv(fn)
            generations[GEN_COL] = [process(txt) if type(txt) == str else " " for txt in generations[GEN_COL]]

            print(generations[:10])

            results = compute_metrics(generations, prompt_col=args.prompt_col, cand_col=GEN_COL, refs_col='rot')

            typ = "greedy"
            if 'beams3' in fn:
                typ = 'beam'
            elif 'p0.9' in fn:
                typ = 'p=0.9'

            size = 'full'
            if '5k' in fn:
                size = '05k'
            elif '10k' in fn:
                size = '10k'
            elif '18k' in fn:
                size = '18k'
            elif '1k' in fn:
                size = '01k'

            model = 'OTHER'
            if 'bart' in fn:
                model = 'bart'
            elif 'gpt' in fn:
                model = 'gpt'
            elif 't5' in fn:
                model = 't5'

            data[i] = results
            data[i]['decoding'] = typ
            data[i]['train_size'] = size
            data[i]['model'] = model
            data[i]['fn'] = fn

            print(data)

            with open(result_path, 'w') as f:
                json.dump(results, f, indent=4)
            i += 1

    out = pd.DataFrame().from_dict(data, orient='index').sort_values(['model', 'decoding', 'train_size'])

    print(out)
    out.to_csv(args.output)
