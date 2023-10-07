# modified based on generate_sentence.py

import argparse
import json
import logging
import os
import sys

import transformers
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, BertForTokenClassification,
                          AutoModelForSeq2SeqLM, DataCollatorForTokenClassification, HfArgumentParser,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer, TrainerCallback)
from transformers.trainer_utils import is_main_process
from strategy_trainer import Seq2SeqTrainer

from data.Datareader import GenerateDataset2 as BartDataset, get_stratege, fix_random

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from transformers import BertTokenizer, BartTokenizer, BartModel, BartConfig, GPT2Tokenizer, BlenderbotSmallTokenizer

from MODEL.MultiSource import BART_MODEL
from generate_sentence import compute_metrics

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
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--warmup_ratio", default=0.7, type=float)
parser.add_argument("--max_source_length", default=512, type=int)
parser.add_argument("--generation_max_length", default=64, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--save_total_limit", type=int, default=3)
parser.add_argument('--metric_for_best_model', default="Bleu_4", type=str)
parser.add_argument('--greater_is_better', default=True, type=bool)
parser.add_argument("--num_train_epochs", default=5, type=int)
parser.add_argument("--evaluation_strategy", default="steps", type=str)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--save_strategy", default="steps", type=str)
parser.add_argument("--load_best_model_at_end", default=True)
parser.add_argument("--ignore_pad_token_for_loss", default=True)
parser.add_argument("--predict_with_generate", default=True)
parser.add_argument("--logging_steps", default=100, type=int)
parser.add_argument("--eval_steps", default=100, type=int)
parser.add_argument("--save_steps", default=100, type=int)

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


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(None, None),
)
save_path = os.path.join(args.model_name_or_path, "inference_results_test")
if not os.path.exists(save_path):
    os.mkdir(save_path)

#### beam=4 prediction
if not os.path.exists(os.path.join(save_path, "MultiESC_test_metrics_beam4.json")):
    fix_random(args.seed)
    test_dataset = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
                               max_target_len=max_target_length, with_strategy=args.with_strategy,
                               sentence_num=args.sen_num, add_cause=args.with_cause, lookahead=args.lookahead)

    predictions = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_target_length, num_beams=4)
    print("beam=4, predict_metrics: ", predictions.metrics)
    json.dump(predictions.metrics, open(os.path.join(save_path, "MultiESC_test_metrics_beam4.json"), "w"), indent=2)
    pred2file = open(os.path.join(save_path, "MultiESC_test_predictions_beam4.txt"), "w")
    for prediction in predictions.predictions[0]:
        decoded_responses = tokenizer.decode(prediction, skip_special_tokens=True)
        pred2file.write(f"{decoded_responses.strip()}\n")
    pred2file.close()

# #### beam=1 prediction
if not os.path.exists(os.path.join(save_path, "MultiESC_test_metrics_beam1.json")):
    fix_random(args.seed)
    test_dataset = BartDataset(args.data_type, args.test_file, tokenizer, max_source_len=args.max_source_length,
                               max_target_len=max_target_length, with_strategy=args.with_strategy,
                               sentence_num=args.sen_num, add_cause=args.with_cause, lookahead=args.lookahead)

    predictions2 = trainer.predict(test_dataset, metric_key_prefix="predict", max_length=max_target_length, num_beams=1)
    print("beam=1, predict_metrics: ", predictions2.metrics)
    json.dump(predictions2.metrics, open(os.path.join(save_path, "MultiESC_test_metrics_beam1.json"), "w"), indent=2)
    #### save the prediction
    pred2file2 = open(os.path.join(save_path, "MultiESC_test_predictions_beam1.txt"), "w")
    for prediction in predictions2.predictions[0]:
        decoded_responses = tokenizer.decode(prediction, skip_special_tokens=True)
        pred2file2.write(f"{decoded_responses.strip()}\n")
    pred2file2.close()



