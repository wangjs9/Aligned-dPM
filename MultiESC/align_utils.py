import copy
import os
import random
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from transformers import BartTokenizer
from data.Datareader import EmotionalIndex, load_json, norm_strategy, _norm


def get_conv_context(tokenizer, data_path='data/train.txt'):
    data = load_json(data_path)
    outputs = []
    for case_example in data:
        dialog = case_example['dialog']
        context = []
        for index, tmp_dic in enumerate(dialog):
            text = tokenizer.decode(tokenizer.encode(tmp_dic["text"]), skip_special_tokens=True)
            if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                res = {
                    "context": context.copy()[-3:],
                    "response": text,
                }
                outputs.append(res)
            context = context + [text]

    with open('data/conv_context.json', 'w') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2, sort_keys=False)
    return outputs


class AlignDataset2(Dataset):
    def __init__(self, model_type, file_path, tokenizer: BartTokenizer, strategy2id=None, max_source_len=510,
                 max_target_len=4, each_sentence_length=32, sentence_num=64, add_cause=False, with_strategy=False,
                 candidate_dir=None, preference_mark=None):
        super(AlignDataset2, self).__init__()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.tokenizer = tokenizer
        self.emotion_index = EmotionalIndex(tokenizer)
        self.each_length = each_sentence_length
        self.sentence_num = sentence_num
        self.strategy2id = strategy2id
        self.model_type = model_type
        self.function_dict = {
            0: self.get_norm_bert_input,
            1: self.get_hierarchical_bert_input,
            2: self.get_norm_strategy_generate_input,
            3: self.get_hierarchical_strategy_generate_input,
            4: self.get_norm_sentence_generate_input,
            5: self.get_hierarchical_sentence_generate_input,
            6: self.get_hierarchical_sentence_generate_input2,
            7: self.get_sequicity_sentence_generate_input,
            8: self.get_hierarchical_sentence_generate_input_add_emotion,
            9: self.get_norm_gpt_generate_input,
        }
        with open(os.path.join(candidate_dir, "candidates.txt"), 'r', encoding='UTF-8') as file:
            candidates = [json.loads(line)["responses"] for line in file]
        candidate_scores = np.load(
            os.path.join(candidate_dir, f"preference_score_{preference_mark}.npy"), allow_pickle=True)
        all_candidates = self.sort_candidate(candidates, candidate_scores)
        self.max_cand_num = max([len(x) for x in all_candidates])
        data = load_json(file_path)
        self.sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token is not None else " "
        self.total_data = []
        self.is_train = 'train' in file_path

        candidate_num = 0
        for case_example in data:
            dialog = case_example['dialog']
            dialog_len = len(dialog)
            emotion_type = case_example['emotion_type']
            problem_type = case_example['problem_type']
            situation = case_example['situation']

            # history = [_norm(emotion_type+' '+problem_type)]
            tot_strategy = []
            for index, tmp_dic in enumerate(dialog):
                if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                    tot_strategy.append(norm_strategy(tmp_dic['strategy']))
            history = [_norm(emotion_type) + self.sep_token + _norm(
                problem_type) + self.sep_token + _norm(situation)]

            tmp_strategy_list = []
            # vad_list = [np.zeros(3)]

            for index, tmp_dic in enumerate(dialog):
                text = _norm(tmp_dic['text'])
                if index == 0 and tmp_dic['speaker'] != 'sys':
                    # vad_list[0] = np.array(tmp_dic['vad'])
                    history[0] = text + self.sep_token + history[0]
                    continue
                if tmp_dic['speaker'] == 'sys' and tmp_dic['strategy'] != "Others":
                    cands = all_candidates[candidate_num]
                    candidate_num += 1
                    # if len(cands) <= 1:
                    #     continue
                    tmp_stratege = norm_strategy(tmp_dic['strategy'])
                    save_s = [x for x in tot_strategy[len(tmp_strategy_list):]].copy()
                    assert len(save_s) > 0, print(tot_strategy, tmp_strategy_list)
                    tmp_history = copy.deepcopy(history)
                    response = text
                    if with_strategy and self.model_type > 4:
                        if self.model_type == 8:
                            tmp_history.append(tmp_stratege)
                            # response = tmp_stratege + " " + text
                        else:
                            tmp_history[-1] = tmp_history[-1] + self.sep_token + tmp_stratege
                    self.total_data.append({
                        "history": tmp_history,
                        "strategy": tmp_stratege,
                        "history_strategy": tmp_strategy_list,
                        "response": response,
                        "future_strategy": ' '.join(save_s),
                        "stage": 5 * index // dialog_len,
                        "candidates": cands,
                        # 'vad': vad_list.copy(),
                    })
                    tmp_strategy_list.append(tmp_stratege)
                if tmp_dic['speaker'] == 'sys':
                    tmp_stratege = norm_strategy(tmp_dic['strategy'])
                    # vad_list.append(np.zeros(3))
                    if with_strategy:
                        tmp_sen = tmp_stratege + self.sep_token + text
                        history.append(tmp_sen)
                    else:
                        history.append(text)
                else:
                    # vad_list.append(np.array(tmp_dic['vad']))
                    if add_cause:
                        cause = tmp_dic['cause']
                        if cause is not None:
                            history.append(cause + self.sep_token + text)
                    else:
                        history.append(text)
        assert candidate_num == len(all_candidates)
        # assert len(self.total_data) == len(all_candidates), print(len(self.total_data), len(all_candidates))
        x = random.randint(1, 50)
        print(x)
        xx = self.__getitem__(x)
        # self.total_data = self.total_data[:100]

        if len(xx['input_ids'].size()) < 2:
            # print(xx)
            print(self.tokenizer.decode(xx['input_ids']))
            print("ans: ", self.tokenizer.decode(xx['labels']))
        if "history_ids" in xx.keys():
            for index in range(len(xx['history_ids'])):
                print(self.tokenizer.convert_ids_to_tokens(xx['history_ids'][index]))
                if 'vads' in xx:
                    print(xx['vads'][index])

    def sort_candidate(self, candidates, candidate_scores):
        assert len(candidate_scores) == len(candidates)
        golden = [gg[0] for gg in candidates]
        cands = [{r: s for r, s in zip(rlst[1:], slst[1:]) if s > 1e-7} for rlst, slst in
                 zip(candidates, candidate_scores)]
        cands = [[gg] + [x[0] for x in sorted(cand.items(), key=lambda x: x[1])] for gg, cand in
                 zip(golden, cands)]

        return cands

    def generate_truncat(self, history, max_len, flag='no_label'):
        input_ids = []
        for text in history:
            content = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(content)
            if self.tokenizer.sep_token_id is not None:
                input_ids.append(self.tokenizer.sep_token_id)
        input_ids[-1] = self.tokenizer.eos_token_id
        if flag == 'label':
            # input_ids.append(self.tokenizer.eos_token_id)
            input_ids = input_ids[:max_len]
            input_ids[-1] = self.tokenizer.eos_token_id
        else:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            # input_ids.append(self.tokenizer.eos_token_id)
            input_ids = input_ids[-max_len:]
            input_ids[0] = self.tokenizer.bos_token_id
        return self.padding_sentence(input_ids, max_len)

    def predict_truncat(self, history, max_len):
        input_ids = []
        for text in history:
            content = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(content)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = [self.tokenizer.cls_token_id] + input_ids
        input_ids = input_ids[:max_len]
        return self.padding_sentence(input_ids, max_len)

    def padding_sentence(self, input_list, max_len):
        return input_list + [self.tokenizer.pad_token_id] * max(max_len - len(input_list), 0)

    def padding_vad(self, vad_list, max_len):
        return np.concatenate([vad_list, np.zeros((max(max_len - len(vad_list), 0), 3))], 0)

    def get_history_tensor(self, history):
        tensor_history = []
        for sentence in history:
            if self.model_type < 2:
                tensor_history.append(torch.tensor(self.predict_truncat([sentence], max_len=self.each_length)).int())
            else:
                tensor_history.append(torch.tensor(self.generate_truncat([sentence], max_len=self.each_length)).int())
        tensor_history = tensor_history[-self.sentence_num:]
        for i in range(max(0, self.sentence_num - len(tensor_history))):
            tensor_history.append(torch.fill_(torch.zeros(self.each_length), self.tokenizer.pad_token_id).int())
        ans = torch.cat([tmp_t.unsqueeze(0) for tmp_t in tensor_history], dim=0)
        return ans, ans != self.tokenizer.pad_token_id

    def get_str_form(self, tmp_str, tmp_len):
        str_list = tmp_str.split()[:tmp_len]
        return ' '.join(str_list)

    def get_norm_bert_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        input_ids = torch.tensor(self.predict_truncat(tmp_history, self.max_source_len))
        labels = torch.tensor(self.strategy2id[tmp_dic['strategy']], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    # 需要加user state.
    def get_hierarchical_bert_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        assert len(tmp_history) % 2 == 1, print(tmp_history)
        combine_history = []
        for i in range(len(tmp_history) // 2):
            tmp_str = self.get_str_form(tmp_history[i * 2],
                                        self.each_length // 2) + ' ' + self.sep_token + ' ' + self.get_str_form(
                tmp_history[i * 2 + 1], self.each_length // 2)
            combine_history.append(tmp_str)
        combine_history.append(self.get_str_form(tmp_history[-1], self.each_length))
        assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        input_ids = torch.tensor(self.predict_truncat(tmp_history, self.max_source_len))
        labels = torch.tensor(self.strategy2id[tmp_dic['strategy']], dtype=torch.long)
        vads = torch.tensor(self.padding_vad(tmp_dic['vad'], self.sentence_num))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_mask": history_mask,
            "vad": vads,
            "labels": labels
        }

    def get_norm_strategy_generate_input(self, tmp_dic):
        tmp_history = tmp_dic['history']
        # assert len(tmp_history) % 2 == 1, print(tmp_history)
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['future_strategy']], self.max_target_len, flag='label'),
                              dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    def get_hierarchical_strategy_generate_input(self, tmp_dic):
        combine_history = tmp_dic['history']
        # assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        # vads = torch.tensor(self.padding_vad(tmp_dic['vad'][:self.sentence_num], self.sentence_num))
        vads = []
        for sentence_id in history_ids:
            sentence = self.tokenizer.convert_ids_to_tokens(sentence_id)
            assert len(sentence) == len(sentence_id), print(sentence, sentence_id, len(sentence), len(sentence_id))
            vads.append(np.array(self.emotion_index.get_one_sentence(sentence)))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['future_strategy']], self.max_target_len, flag='label'),
                              dtype=torch.long)  # print('history_ids', history_ids.size())
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            "vads": torch.tensor(vads),
        }

    def get_norm_sentence_generate_input(self, tmp_dic):
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='label'),
                              dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    def get_norm_gpt_generate_input(self, tmp_dic):
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(
            self.generate_truncat([tmp_dic['strategy'] + ' ' + tmp_dic['response']], self.max_target_len, flag='label'),
            dtype=torch.long)
        if not self.is_train:
            return {
                "input_ids": input_ids,
                "attention_mask": input_ids != self.tokenizer.pad_token_id,
                "labels": torch.cat((input_ids, labels))
            }
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": input_ids
        }

    def get_hierarchical_sentence_generate_input(self, tmp_dic):
        combine_history = tmp_dic['history']
        # combine_history = tmp_history
        # assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        # assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        # input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='label'),
                              dtype=torch.long)
        return {
            "input_ids": history_ids,
            "attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            # "vad": vads,
        }

    def get_hierarchical_sentence_generate_input2(self, tmp_dic):
        combine_history = tmp_dic['history']
        # combine_history = tmp_history
        # assert len(tmp_dic['vad']) == len(combine_history), print(len(tmp_dic['vad']), len(combine_history))
        history_ids, history_mask = self.get_history_tensor(combine_history)
        # vads = torch.tensor(self.padding_vad(tmp_dic['vad'][:self.sentence_num], self.sentence_num))
        # assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='label'),
                              dtype=torch.long)
        # print('history_ids', history_ids.size())
        # print('vads', vads.size())
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            # "vad": vads,
        }

    def get_sequicity_sentence_generate_input(self, tmp_dic):
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(
            self.generate_truncat([tmp_dic['strategy'] + ' ' + tmp_dic['response']], self.max_target_len, flag='label'),
            dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "labels": labels
        }

    def get_hierarchical_sentence_generate_input_add_emotion(self, tmp_dic):
        combine_history = tmp_dic['history']
        history_ids, history_mask = self.get_history_tensor(combine_history)
        # vads = torch.tensor(self.padding_vad(tmp_dic['vad'][:self.sentence_num], self.sentence_num))
        vads = []
        for sentence_id in history_ids:
            sentence = self.tokenizer.convert_ids_to_tokens(sentence_id)
            assert len(sentence) == len(sentence_id), print(sentence, sentence_id, len(sentence), len(sentence_id))
            vads.append(np.array(self.emotion_index.get_one_sentence(sentence)))
        assert len(vads) == len(history_ids), print(len(vads), len(history_ids))
        # vads = torch.tensor(vads)
        input_ids = torch.tensor(self.generate_truncat(tmp_dic['history'], self.max_source_len))
        labels = torch.tensor(self.generate_truncat([tmp_dic['response']], self.max_target_len, flag='label'),
                              dtype=torch.long)
        # candidate_ids = self.tokenizer.batch_encode_plus(tmp_dic["candidates"], return_tensors="pt",
        #                                                  padding="max_length", truncation="longest_first",
        #                                                  max_length=self.max_target_len + 1)["input_ids"][:, 1:]
        valid_candidate_ids = self.tokenizer.batch_encode_plus(tmp_dic["candidates"], return_tensors="pt",
                                                               padding="max_length", truncation="longest_first",
                                                               max_length=self.max_target_len + 1)["input_ids"][:, 1:]
        candidate_ids = torch.ones((self.max_cand_num, self.max_target_len), dtype=torch.long)
        candidate_ids[:valid_candidate_ids.size(0), :] = valid_candidate_ids
        candidate_ids[0] = labels
        # print('history_ids', history_ids.size())
        # print('vads', vads.size())
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
            "history_ids": history_ids,
            "history_attention_mask": history_ids != self.tokenizer.pad_token_id,
            "labels": labels,
            "vads": torch.tensor(vads),
            "candidate_ids": candidate_ids,
        }

    def see_one_item(self, item):
        xx = self.__getitem__(item)
        if "history_ids" in xx:
            for x in zip(xx['history_ids'], xx['history_attention_mask']):
                print(self.tokenizer.decode(x[0]))
        print(self.tokenizer.decode(xx['labels']))

    def __getitem__(self, item):
        tmp_dic = self.total_data[item]
        # print(tmp_dic)
        return self.function_dict[self.model_type](tmp_dic)

    def __len__(self):
        return len(self.total_data)
