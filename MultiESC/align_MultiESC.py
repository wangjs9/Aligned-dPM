# modified based on generate_sentence.py

import argparse
import json
import logging
import os
import sys
import time
import datetime
from collections import defaultdict
import numpy as np
from functools import partial

from torch.optim.lr_scheduler import LambdaLR
import transformers
from transformers import (HfArgumentParser, Seq2SeqTrainingArguments, BartTokenizer, GPT2Tokenizer,
                          BlenderbotSmallTokenizer)
from transformers.trainer_utils import is_main_process
from strategy_trainer import Seq2SeqTrainer

from data.Datareader import GenerateDataset2 as BartDataset, get_stratege, fix_random

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from MODEL.MultiSource import BART_MODEL
from generate_sentence import compute_metrics, get_optimer

from align_utils import AlignDataset2
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default='facebook/bart-base', type=str)
parser.add_argument("--lr2", default=1e-4, type=float)
parser.add_argument("--do_train", default=True)
parser.add_argument("--do_eval", default=True)
parser.add_argument("--do_predict", default=True)
parser.add_argument("--train_file", default="./data/train.txt", type=str)
parser.add_argument("--validation_file", default="./data/valid.txt", type=str)
parser.add_argument("--test_file", default="./data/test.txt", type=str)
parser.add_argument("--output_dir", default="./aligned_output/", type=str)
parser.add_argument("--per_device_train_batch_size", default=8, type=int)
parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--warmup_ratio", default=0.1, type=float)
# parser.add_argument("--warmup_steps", default=800, type=int)
parser.add_argument("--max_source_length", default=512, type=int)
parser.add_argument("--generation_max_length", default=64, type=int)
parser.add_argument("--seed", default=3407, type=int)
parser.add_argument("--save_total_limit", type=int, default=3)
parser.add_argument('--metric_for_best_model', default="Bleu_4", type=str)
parser.add_argument('--greater_is_better', default=True, type=bool)
parser.add_argument("--num_train_epochs", default=10, type=int)
parser.add_argument("--evaluation_strategy", default="steps", type=str)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--save_strategy", default="steps", type=str)
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--ignore_pad_token_for_loss", default=True)
parser.add_argument("--predict_with_generate", default=True)
parser.add_argument("--logging_steps", default=200, type=int)
parser.add_argument("--eval_steps", default=200, type=int)
parser.add_argument("--save_steps", default=200, type=int)

parser.add_argument("--data_type", default=4, type=int)
parser.add_argument("--model_type", default=1, type=int)
parser.add_argument("--sen_num", default=64, type=int)
parser.add_argument("--with_cause", action="store_true")
parser.add_argument("--lookahead", action="store_true")
parser.add_argument("--not_pretrain", action="store_true")
parser.add_argument("--config_path", default='../../MODEL/transformer_config', type=str)

parser.add_argument("--with_strategy", action="store_true")
parser.add_argument("--preference_model_dir", type=str, default=None)
parser.add_argument("--candidate_num", default=10, type=int, help="number of samples for calibration")

args = parser.parse_args()
fix_random(args.seed)
arg_dict = args.__dict__
print(arg_dict)
logger = logging.getLogger(__name__)

train_parser = HfArgumentParser(Seq2SeqTrainingArguments)
print("args.model_name_or_path: ", args.model_name_or_path)


def set_log(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


###################
# Dataset and model ready
###################
strategys = get_stratege('data/new_strategy.json', norm=True)
strategy_list = [v for k, v in enumerate(strategys)]
BartForConditionalGeneration = BART_MODEL[args.model_type]
if args.model_type == 3:
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(args.model_name_or_path)
elif args.model_type == 4:
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token_id = tokenizer.unk_token_id
    # tokenizer.sep_token = "[SEP]"
else:
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

tokenizer.add_tokens(strategy_list)


def test_checkpoints(aligned_model_dir, args):
    checkpoint_list = os.listdir(aligned_model_dir)
    checkpoint_list = [file for file in checkpoint_list if file.startswith('checkpoint-')]

    training_args = train_parser.parse_dict(vars(args))[0]
    max_target_length = args.generation_max_length
    test_dataset = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
                               max_target_len=max_target_length, with_strategy=args.with_strategy,
                               sentence_num=args.sen_num, add_cause=args.with_cause, lookahead=args.lookahead)

    for load_checkpoint_path in checkpoint_list:
        load_checkpoint_path = os.path.join(aligned_model_dir, load_checkpoint_path)
        save_path = os.path.join(load_checkpoint_path, "inference_results")
        os.makedirs(save_path, exist_ok=True)
        model, loading_info = BartForConditionalGeneration. \
            from_pretrained(load_checkpoint_path, output_loading_info=True)

        if args.model_type == 4:
            model.config.pad_token_id = tokenizer.unk_token_id
        model.resize_token_embeddings(len(tokenizer))
        model.config.max_length = args.generation_max_length

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            optimizers=(None, None),
        )

        #### beam=4 prediction
        # if not os.path.exists(os.path.join(save_path, "aligned_test_metrics_beam4.json")):
        predictions = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_target_length,
                                      num_beams=4)
        print("beam=4, predict_metrics: ", predictions.metrics)
        json.dump(predictions.metrics, open(os.path.join(save_path, "aligned_test_metrics_beam4.json"), "w"),
                  indent=2)
        pred2file = open(os.path.join(save_path, "aligned_test_predictions_beam4.txt"), "w")
        for pred in predictions.predictions[0]:
            decoded_responses = tokenizer.decode(pred, skip_special_tokens=True)
            pred2file.write(f"{decoded_responses.strip()}\n")
        pred2file.close()

        #### beam=1 prediction
        # if not os.path.exists(os.path.join(save_path, "aligned_test_metrics_beam1.json")):
        predictions2 = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_target_length,
                                       num_beams=1)
        print("beam=1, predict_metrics: ", predictions2.metrics)
        json.dump(predictions2.metrics, open(os.path.join(save_path, "aligned_test_metrics_beam1.json"), "w"),
                  indent=2)
        pred2file = open(os.path.join(save_path, "aligned_test_predictions_beam1.txt"), "w")
        for pred in predictions2.predictions[0]:
            decoded_responses = tokenizer.decode(pred, skip_special_tokens=True)
            pred2file.write(f"{decoded_responses.strip()}\n")
        pred2file.close()


def baseline_check(args):
    training_args = train_parser.parse_dict(vars(args))[0]
    set_log(training_args)

    model, loading_info = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        output_loading_info=True
    )
    if args.model_type == 4:
        model.config.pad_token_id = tokenizer.unk_token_id

    model.resize_token_embeddings(len(tokenizer))
    model.config.max_length = args.generation_max_length

    max_target_length = args.generation_max_length
    train_dataset = BartDataset(args.data_type, args.train_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    valid_dataset = BartDataset(args.data_type, args.validation_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    test_dataset = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
                               max_target_len=max_target_length, with_strategy=args.with_strategy,
                               sentence_num=args.sen_num, add_cause=args.with_cause, lookahead=args.lookahead)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(None, None),
    )
    save_path = os.path.join(args.model_name_or_path, "inference_results")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    #### beam=4 valid
    if not os.path.exists(os.path.join(save_path, "MultiESC_valid_metrics_beam4.json")):
        predictions = trainer.predict(valid_dataset, metric_key_prefix="valid", max_length=max_target_length,
                                      num_beams=4)
        print("beam=4, valid_metrics: ", predictions.metrics)
        json.dump(predictions.metrics, open(os.path.join(save_path, "MultiESC_valid_metrics_beam4.json"), "w"),
                  indent=2)
        pred2file_valid = open(os.path.join(save_path, "MultiESC_valid_predictions_beam4.txt"), "w")
        for pred in predictions.predictions[0]:
            decoded_responses = tokenizer.decode(pred, skip_special_tokens=True)
            pred2file_valid.write(f"{decoded_responses.strip()}\n")
            pred2file_valid.close()
        pred2file_valid.close()

    #### beam=1 valid
    if not os.path.exists(os.path.join(save_path, "MultiESC_valid_metrics_beam1.json")):
        predictions2 = trainer.predict(valid_dataset, metric_key_prefix="valid", max_length=max_target_length,
                                       num_beams=1)
        print("beam=1, valid_metrics: ", predictions2.metrics)
        json.dump(predictions2.metrics, open(os.path.join(save_path, "MultiESC_valid_metrics_beam1.json"), "w"),
                  indent=2)
        pred2file_valid = open(os.path.join(save_path, "MultiESC_valid_predictions_beam1.txt"), "w")
        for pred in predictions2.predictions[0]:
            decoded_responses = tokenizer.decode(pred, skip_special_tokens=True)
            pred2file_valid.write(f"{decoded_responses.strip()}\n")
        pred2file_valid.close()

    #### beam=4 prediction
    if not os.path.exists(os.path.join(save_path, "MultiESC_test_metrics_beam4.json")):
        predictions = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_target_length,
                                      num_beams=4)
        print("beam=4, predict_metrics: ", predictions.metrics)
        json.dump(predictions.metrics, open(os.path.join(save_path, "MultiESC_test_metrics_beam4.json"), "w"), indent=2)
        #### save the prediction
        pred2file = open(os.path.join(save_path, "MultiESC_test_predictions_beam4.txt"), "w")
        for prediction in predictions.predictions[0]:
            decoded_responses = tokenizer.decode(prediction, skip_special_tokens=True)
            pred2file.write(f"{decoded_responses.strip()}\n")
        pred2file.close()

    #### beam=1 prediction
    if not os.path.exists(os.path.join(save_path, "MultiESC_test_metrics_beam1.json")):
        predictions2 = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_target_length,
                                       num_beams=1)
        print("beam=1, predict_metrics: ", predictions2.metrics)
        json.dump(predictions2.metrics, open(os.path.join(save_path, "MultiESC_test_metrics_beam1.json"), "w"),
                  indent=2)
        #### save the prediction
        pred2file2 = open(os.path.join(save_path, "MultiESC_test_predictions_beam1.txt"), "w")
        for prediction in predictions2.predictions[0]:
            decoded_responses = tokenizer.decode(prediction, skip_special_tokens=True)
            pred2file2.write(f"{decoded_responses.strip()}\n")
        pred2file2.close()


def _get_schedule_with_warmup(current_step: int, *, num_warmup_steps: int):
    return min((current_step + 1e-8) ** (-0.5), current_step * ((num_warmup_steps + 1e-8) ** (-1.5)))


def get_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases from the max_lr set in the optimizer, after
    a warmup period during which it increases from 0 to the max_lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        max_lr (`float`):
            The maximum learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_lambda = partial(
        _get_schedule_with_warmup,
        num_warmup_steps=num_warmup_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def alignment(args):
    preference_model_args = argparse.ArgumentParser()
    preference_model_args_dict = vars(preference_model_args)
    with open(os.path.join(args.preference_model_dir, "args.json"), "r") as f:
        summary_dict = json.load(f)
        preference_model_args_dict.update(summary_dict)
    args.mode = preference_model_args.mode

    args.output_dir = os.path.join(
        args.output_dir,
        f"MultiESC_align_{args.mode}_seed{args.seed}_{datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')}")
    assert not os.path.exists(args.output_dir), print(f"{args.output_dir} already exists!")
    os.mkdir(args.output_dir)

    training_args = train_parser.parse_dict(vars(args))[0]
    set_log(training_args)

    model, loading_info = BartForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        output_loading_info=True
    )
    second_parameters = loading_info['missing_keys']
    if args.model_type == 4:
        model.config.pad_token_id = tokenizer.unk_token_id

    model.resize_token_embeddings(len(tokenizer))
    model.config.max_length = args.generation_max_length

    assert isinstance(args.with_strategy, bool), print("with_strategy's type is: ", type(args.with_strategy))
    max_target_length = args.generation_max_length
    valid_dataset = BartDataset(args.data_type, args.validation_file, tokenizer, max_source_len=args.max_source_length,
                                max_target_len=max_target_length, with_strategy=args.with_strategy,
                                sentence_num=args.sen_num, add_cause=args.with_cause)
    test_dataset = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
                               max_target_len=max_target_length, with_strategy=args.with_strategy,
                               sentence_num=args.sen_num, add_cause=args.with_cause, lookahead=args.lookahead)
    print(len(valid_dataset), len(test_dataset))
    my_optim = get_optimer(model, second_parameters, training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        # eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(my_optim, None),
    )

    #########################################################################
    # sampling candidates and computing preference scores
    #########################################################################
    args.candidate_dir = args.model_name_or_path + f"_candidate_{args.candidate_num}"
    if not os.path.exists(args.candidate_dir):
        os.mkdir(args.candidate_dir)

    if not os.path.exists("data/conv_context.json"):
        logger.info("getting conversation context...")
        from align_utils import get_conv_context
        conv_context = get_conv_context(tokenizer)
    else:
        conv_context = json.load(open("data/conv_context.json", "r"))

    if not os.path.exists(os.path.join(args.candidate_dir, f"candidates.txt")):
        train_dataset = BartDataset(args.data_type, args.train_file, tokenizer, max_source_len=args.max_source_length,
                                    max_target_len=max_target_length, with_strategy=args.with_strategy,
                                    sentence_num=args.sen_num, add_cause=args.with_cause)
        logger.info(f"sampling {args.candidate_num} candidates...")
        trainer.candidate_sampling(train_dataset, tokenizer, conv_context, candidate_num=args.candidate_num,
                                   save_dir=args.candidate_dir)
    preference_mark = args.preference_model_dir.split('/')[-1]
    if not os.path.exists(os.path.join(args.candidate_dir, f"preference_score_{preference_mark}.npy")):
        from preference_modeling import compute_scores
        dataloader_kwargs = {
            "max_input_length": 160,
            "max_decoder_input_length": 40,
            "only_encode": False,
        }
        logger.info(f"computing preference scores, preference model: {args.mode}...")
        temp_args = argparse.ArgumentParser()
        temp_args_dict = vars(temp_args)
        temp_args_dict.update({
            "device": training_args.device,
            "n_gpu": training_args.n_gpu,
            "local_rank": training_args.local_rank,
            "eval_batch_size": 32,
            "preference_model_dir": args.preference_model_dir,
            "candidate_dir": args.candidate_dir,
        })

        compute_scores(preference_model_args, temp_args, **dataloader_kwargs)
        del temp_args

    #########################################################################
    # aligning MultiESC with preference scores
    #########################################################################

    align_dataset = AlignDataset2(
        args.data_type,
        args.train_file,
        tokenizer,
        max_source_len=args.max_source_length,
        max_target_len=max_target_length,
        with_strategy=args.with_strategy,
        sentence_num=args.sen_num,
        add_cause=args.with_cause,
        candidate_dir=args.candidate_dir,
        preference_mark=preference_mark,
    )
    logger.info(f"the length of the dataset is {len(align_dataset)}.")

    trainer.train_dataset = align_dataset
    json.dump(arg_dict, open(os.path.join(args.output_dir, "args.json"), "w"), indent=2)
    logger.info(f"Aligning *MultiESC* with *{args.mode}*...")
    train_result = trainer.align_base_model()
    trainer.save_model(output_dir=args.output_dir)

    metrics = train_result.metrics
    predict_metrics = trainer.evaluate(test_dataset, metric_key_prefix="predict", max_length=max_target_length,
                                       num_beams=4)

    predict_metrics2 = trainer.evaluate(test_dataset, metric_key_prefix="predict", max_length=max_target_length,
                                        num_beams=1)
    print("beam=4, predict_metrics: ", predict_metrics)
    print("beam=1, predict_metrics: ", predict_metrics2)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return predict_metrics, predict_metrics2


if __name__ == "__main__":
    start_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    metric1, metric4 = defaultdict(list), defaultdict(list)

    if not os.path.exists(args.output_dir):
        print("new a _dir: ", args.output_dir)
        os.makedirs(args.output_dir)

    beam4, beam1 = alignment(args)
    for k in beam1.keys():
        metric1[k].append(beam1[k])
        metric4[k].append(beam4[k])
    for k in metric1.keys():
        print(f"beam1_{k}", metric1[k], "mean: ", np.mean(metric1[k]), "std: ", np.std(metric1[k]))
    for k in metric1.keys():
        print(f"beam4_{k}", metric4[k], "mean: ", np.mean(metric4[k]), "std: ", np.std(metric4[k]))
    end_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    print(start_time, end_time)
    baseline_check(args)
    # test_checkpoints(args.output_dir, args)
