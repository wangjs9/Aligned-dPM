import pandas as pd
import argparse
from datasets import load_dataset, load_metric, Dataset
import pandas as pd
from collections import OrderedDict, Counter
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union
from transformers.trainer import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForSeq2Seq, AutoTokenizer, \
    DataCollatorForLanguageModeling, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, Trainer, BatchEncoding
import nltk, os, csv, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader, IterableDataset
from tqdm import tqdm
import json, sys
import math
import collections
import warnings

warnings.filterwarnings("ignore")

EncodedInput = List[int]

ATTR_MAPPINGS = {
    'AGREE_TO_STR': OrderedDict([(1, "nobody"), (2, "rare"), (3, "controversial"), (4, "most"), (5, "all")]),
    'VIOLATION_SEVERITY_TO_STR': OrderedDict([(1, "fine"), (2, "unwise"), (3, "bad"), (4, "horrible"), (5, "worst")]),
    'ALIGNMENT_TO_STR': OrderedDict([(0, "disagrees"), (1, "neutral"), (2, "agrees")])
}

from transformers.trainer_pt_utils import find_batch_size, DistributedTensorGatherer, SequentialDistributedSampler, \
    nested_concat, nested_numpify, nested_truncate, IterableDatasetShard
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, denumpify_detensorize, speed_metrics
from transformers.deepspeed import deepspeed_init
from transformers.utils import logging

logger = logging.get_logger(__name__)


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_all_attributes(df):
    attributes = set()
    for _, row in df.iterrows():
        if 'moral' in row:
            check = 'moral'
        elif 'rot-moral-foundations' in row:
            check = 'rot-moral-foundations'
        else:
            continue
        if pd.notna(row[check]) and not len(row[check]) == 0:
            for category in sorted(row[check].split("|")):
                attributes.add(f"<{category}>")

    for mapping_key in ATTR_MAPPINGS:
        for key in ATTR_MAPPINGS[mapping_key]:
            attributes.add(f"<{ATTR_MAPPINGS[mapping_key][key]}>")
    return list(attributes)


def init_attribute_embeddings(model, tokenizer, special_tokens):
    """
    Initialize each attribute embedding (e.g. <very-bad>) with the bag of words of its words (vec(very) + vec(bad))
    """
    embeddings = model.get_input_embeddings()
    unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    for word in special_tokens:
        index = tokenizer.convert_tokens_to_ids(word)

        if word.startswith("<"):
            other_words = word[1:-1].replace("-", " ").split()
            other_indices = tokenizer.convert_tokens_to_ids(other_words)
            other_indices = [i for i in other_indices if i != unk_token_id]
            if len(other_indices) == 0:
                continue
            elif len(other_indices) == 1:
                vec = embeddings.weight.data[other_indices[0], :]
            else:
                vec = torch.sum(
                    torch.stack([embeddings.weight.data[i, :] for i in other_indices])
                )

            embeddings.weight.data[index, :] = vec

    model.set_input_embeddings(embeddings)


def get_attr(row: pd.Series, col: str) -> List[str]:
    res: List[str] = []

    if col == "moral":
        if pd.notna(row["moral"]) and not len(row["moral"]) == 0:
            # Multi-label
            for category in sorted(row["moral"].split("|")):
                res.append(f"<{category}>")
    elif col == "rot-moral-foundations":
        if pd.notna(row["rot-moral-foundations"]) and not len(row["rot-moral-foundations"]) == 0:
            for category in sorted(row["rot-moral-foundations"].split("|")):
                res.append(f"<{category}>")
    elif col == "rot-agree":
        if pd.notna(row["rot-agree"]):
            res.append(f"<{ATTR_MAPPINGS['AGREE_TO_STR'][row['rot-agree']]}>")
    elif col == "violation-severity":
        if pd.notna(row["violation-severity"]):
            res.append(f"<{ATTR_MAPPINGS['VIOLATION_SEVERITY_TO_STR'][row['violation-severity']]}>")
    elif col == "A_agrees":
        if pd.notna(row['A_agrees']):
            res.append(f"<{ATTR_MAPPINGS['ALIGNMENT_TO_STR'][row['A_agrees']]}>")
    else:
        raise ValueError(f"Unknown attribute: '{col}'")

    return res


def get_rot_attributes(row: pd.Series) -> List[str]:
    """
    Gets a row from the rot-details tsv file and returns a list of string rot-related attributes
    :param row: dataframe row
    :return: a list of string rot-related attributes
    """
    return (
            get_attr(row, "A_agrees")
            + get_attr(row, "rot-agree")
            + get_attr(row, "violation-severity")
            + get_attr(row, "moral")
    )


def build(row, format_string):
    formats = format_string.split("~")
    outputs = []
    for form in formats:

        output = []
        for element in form.split():
            if '[' in element:
                output.append(element)
            elif '<' in element:
                output.append(' '.join(get_attr(row, element.replace('<', '').replace('>', ''))))
            else:
                output.append(row[element].strip())
        outputs.append(" ".join(output))
    outputs[-1] += " <eos>"
    return outputs


def update_build(row, format_string):
    formats = format_string.split("~")
    outputs = []
    for form in formats:

        output = []
        for element in form.split():
            if '[' in element:
                output.append(element)
            elif '<' in element:
                output.append(' '.join(get_attr(row, element.replace('<', '').replace('>', ''))))
            elif type(row[element]) == list:
                output.append([x.strip() for x in row[element]])
            else:
                output.append(row[element].strip())
        if type(output[-1]) != list:
            outputs.append(" ".join(output))
        else:
            outputs.append(output[0])
    if type(outputs[-1]) == list:
        outputs[-1] = [x + " <eos>" for x in outputs[-1]]
    else:
        outputs[-1] += " <eos>"
    return outputs


def tokenize(string, tokenizer, eos_id=None, pad=False):
    if pad:
        return tokenizer(string, padding=True)
    else:
        return tokenizer(string)


#     if not eos_id:
#         eos_id = tokenizer.eos_token_id

#     t = tokenizer(string)
#     input_ids = t['input_ids']
#     if type(input_ids[0])==list:
#         input_ids = [ np.array(row)[np.array(row)!=eos_id] for row in input_ids]
#     else:
#         input_ids = np.array(input_ids)
#         input_ids = input_ids[input_ids!=eos_id]
#     t['input_ids'] = input_ids
#     return t

def preprocess(examples, tokenizer, format_string):
    source_target = [build(row, format_string)
                     for _, row in pd.DataFrame(dict(examples)).iterrows()]
    source = [tup[0] for tup in source_target]
    target = [tup[1] if len(tup) > 1 else "" for tup in source_target]

    model_inputs = tokenize(source, tokenizer)  # tokenizer(source)

    with tokenizer.as_target_tokenizer():
        labels = tokenize(target, tokenizer)  # tokenizer(target)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def rm_eos(lst_of_lst):
    return [lst if sum(lst) > 1 else [0] * len(lst) for lst in lst_of_lst]


def aligned_preprocess(examples, tokenizer, format_string, max_sample_num=10):
    source_target = [update_build(row, format_string)
                     for _, row in pd.DataFrame(dict(examples)).iterrows()]
    source = [tup[0] for tup in source_target]
    target = [tup[1] + [""] * (max_sample_num - len(tup[1])) for tup in source_target]

    model_inputs = tokenize(source, tokenizer)  # tokenizer(source)

    with tokenizer.as_target_tokenizer():
        labels = [tokenize(t, tokenizer, pad=True) for t in target]  # tokenizer(target)

    model_inputs["labels"] = [rm_eos(l["input_ids"]) for l in labels]
    return model_inputs


def decode(args, df, model, tokenizer, skip_special_tokens=True, remove_history=False):
    model = model
    is_greedy = (args.top_p == 0) and (args.top_k == 0) and (args.beams == 0)

    generations = []

    for _, row in df.iterrows():
        input_ids = torch.tensor([row['input_ids']], device='cuda')

        out = model.generate(
            input_ids,
            do_sample=args.beams == 0,
            max_length=args.maxlen,
            temperature=args.temperature,
            top_p=args.top_p if args.top_p > 0 else None,
            top_k=args.top_k if args.top_k > 0 else None,
            num_beams=args.beams if args.beams > 0 else None,
            early_stopping=True,
            # pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=3,
            # eos_token_id=tokenizer.eos_token_id
        )
        if remove_history:
            generations.append(
                tokenizer.decode(out[:, input_ids.shape[-1]:][0], skip_special_tokens=skip_special_tokens))

        else:
            generations.append(tokenizer.decode(out[0], skip_special_tokens=skip_special_tokens))
    return generations


from transformers.trainer import *
from transformers.trainer_seq2seq import *


class AlignedDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            label_num = 1
            if isinstance(labels[0][0], list):
                labels = sum(labels, [])
                label_num = len(features[0]["labels"])
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            # for feature in features:
            new_labels = []
            for idx, label in enumerate(labels):
                remainder = [self.label_pad_token_id] * (max_label_length - len(label))
                if isinstance(label, list):
                    new_labels.append(
                        label + remainder if padding_side == "right" else remainder + label
                    )
                elif padding_side == "right":
                    new_labels.append(np.concatenate([label, remainder]).astype(np.int64))
                else:
                    new_labels.append(np.concatenate([remainder, label]).astype(np.int64))
            new_labels = np.array(new_labels)
            if label_num > 1:
                new_labels = new_labels.reshape(-1, label_num, max_label_length)
            for idx, feature in enumerate(features):
                feature["labels"] = new_labels[idx]

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        if len(features["labels"].size()) > 2:
            features["labels"] = features["labels"].masked_fill(features["labels"] == self.tokenizer.pad_token_id, -100)
        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


class AlignedDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0]["labels"][0], list):
            labels = [torch.tensor(l) for l in sum([example["labels"] for example in examples], [])]
            labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = labels.view(len(examples), len(examples[0]["labels"]), -1)
            for idx, example in enumerate(examples):
                examples[idx]["labels"] = labels[idx].tolist()
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


class SampleSeq2SeqTrainer(Seq2SeqTrainer):

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        assert self._num_beams != None
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "do_sample": False,
            "num_beams": self._num_beams,
            "num_beam_groups": self._num_beams,
            "num_return_sequences": self._num_beams,
            "no_repeat_ngram_size": 3,
            "diversity_penalty": 4.0,
            "length_penalty": -1.0,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, len(all_preds))
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
