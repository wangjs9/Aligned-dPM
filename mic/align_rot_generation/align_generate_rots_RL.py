import torch
from rot_generation.common import *
from torch import Tensor

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from preference_modeling.inputters import inputters
from preference_modeling.utils.building_utils import build_model, deploy_model
import random
import warnings
import nltk

from trl import PPOConfig, AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead
from align_rot_generation.ppo_utils import AlignPPOTrainer as PPOTrainer

warnings.filterwarnings("ignore")


def format_str_to_savefile_name(format_str):
    return '_'.join(
        [x.replace('<', '').replace('>', '') for x in format_str.replace("~", 'TARGET').split() if '[' not in x])


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


class preference_pipe(object):
    def __init__(self, preference_model_args, args, **dataloader_kwargs):
        kwargs = {'mode': preference_model_args.mode}
        self.toker, self.model = build_model(
            checkpoint=os.path.join(args.preference_model_checkpoint, "best.bin"),
            args=preference_model_args,
            **kwargs,
        )
        preference_model_args.n_gpu = 1
        preference_model_args.device = "cuda" if torch.cuda.is_available() else "cpu"
        deploy_model(self.model, preference_model_args)
        self.model.eval()
        self.inputter = inputters["mic"]()
        self.dataloader_kwargs = dataloader_kwargs
        self.dataloader_kwargs.update({'no_bar_info': True})
        self.device = preference_model_args.device

    def compute_score(self, text_batch):
        infer_dataloader = self.inputter.infer_dataloader(
            toker=self.toker,
            corpus_df=text_batch,
            batch_size=1,
            **self.dataloader_kwargs,
        )
        predictions = []
        for batch, sample_idx in infer_dataloader:
            batch = {k: v.to("cuda") if isinstance(v, Tensor) else v for k, v in batch.items()}
            scores = self.model.predict(**batch)[:, -1].cpu()
            predictions.append(scores)
        return predictions


def preprocess(examples, tokenizer, format_string):
    source_target = [build(row, format_string) for _, row in pd.DataFrame(dict(examples)).iterrows()]
    source = [tup[0] for tup in source_target]
    target = [tup[1] if len(tup) > 1 else "" for tup in source_target]

    model_inputs = tokenize(source, tokenizer)  # tokenizer(source)

    with tokenizer.as_target_tokenizer():
        labels = tokenize(target, tokenizer)  # tokenizer(target)

    model_inputs["label"] = labels["input_ids"]
    return model_inputs


def final_compute_metrics(decoded_preds, decoded_labels, metric, metric2, tokenizer):
    # Rouge expects a newline after each sentence
    decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    decoded_labels_expanded = [[x] for x in decoded_labels]
    result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

    # print(result2)
    result['sacrebleu'] = round(result2["score"], 2)

    r = {k: round(v, 4) for k, v in result.items()}

    return r


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str,
                        default='rot_generation/output/t5-small_Q_A_TARGET_rot_epochs5_batch16/model')
    parser.add_argument('--architecture', default='', choices=['seq2seq', 'causal-lm'])
    parser.add_argument('--input', type=str, help='path to input file (MIC dataset)')
    parser.add_argument('--output', type=str, default='results', help='path to directory for outputting results')
    parser.add_argument('--seed', type=int, default=13, help='random seed for replicability')
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--format_string', type=str, default="Q [answ] A [rot] ~ rot", help='how to format the dataset')
    parser.add_argument('--source_name', type=str, default='QA',
                        help='the name of the source column to write in the out file')
    parser.add_argument('--target_name', type=str, default='rot',
                        help='the name of the target column to write in the out file')
    parser.add_argument('--maxlen', type=int, default=20,
                        help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--warmup_steps', type=int, default=8000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)  # 1e-4
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--generate_attributes', action='store_true')
    parser.add_argument('--metric_for_best_model', default="rouge1", type=str)
    parser.add_argument('--greater_is_better', default=True, type=bool)
    parser.add_argument('--train_size', type=int, default=-1,
                        help='the number of train datapoints to use as an ablation (default is to use the full train set)')

    parser.add_argument('--preference_model_checkpoint', type=str,
                        default="../preference_modeling/output/mic_d-PM_23-0528-1653")
    parser.add_argument('--checking_steps', type=int, default=2000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)

    parser.add_argument('--target_kl', type=float, default=0.1)
    parser.add_argument('--kl_penalty', type=str, default="kl")
    parser.add_argument('--use_score_scaling', action='store_true')
    parser.add_argument('--use_score_norm', action='store_true')
    parser.add_argument('--score_clip', type=float, default=None)

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)

    mode = "d-PM" if "d-PM" in args.preference_model_checkpoint else "soft"
    OUT_DIR = os.path.join(
        args.output,
        f"{args.model_checkpoint.split('/')[-2].split('_')[0]}_{format_str_to_savefile_name(args.format_string)}_epochs{args.epochs}_batch{args.batchsize}_seed{args.seed}_{mode}")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # configure for architecture
    config = PPOConfig(
        model_name=args.model_checkpoint,
        learning_rate=args.lr,
        # log_with="wandb",  # use wandb
        mini_batch_size=args.batchsize,
        batch_size=args.batchsize * args.gradient_accumulation_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping=True,
        target_kl=args.target_kl,
        kl_penalty=args.kl_penalty,
        seed=args.seed,
        use_score_scaling=args.use_score_scaling,
        use_score_norm=args.use_score_norm,
        score_clip=args.score_clip,
        # remove_unused_columns=False
    )

    if args.architecture == '':
        if 'gpt' in args.model_checkpoint:
            args.architecture = 'causal-lm'
        else:
            args.architecture = 'seq2seq'
    if args.architecture == 'causal-lm':
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_checkpoint)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_checkpoint)
    else:
        model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_checkpoint)
        ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")

    dataset = load_dataset("csv", data_files={'train': os.path.join(args.input, 'tmp/train.csv'),
                                              # 'train': os.path.join(args.input, 'tmp/dev.csv'),
                                              'validation': os.path.join(args.input, 'tmp/dev.csv')})

    # add special tokens to tokenizer
    special_tokens = list(
        set(
            get_all_attributes(pd.DataFrame(dataset['train']))
            + [
                "[answ]",
                "[attrs]",
                "[rot]",
                "[attrs_and_rot]",
                "<pad>",
                "<eos>"
            ]
        )
    )

    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.add_tokens(special_tokens)
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    init_attribute_embeddings(model.pretrained_model, tokenizer, special_tokens)

    args.pad_token_id = tokenizer.pad_token_id

    tokenize_format_string = args.format_string.replace("~",
                                                        "") if args.architecture == 'causal-lm' else args.format_string
    tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer, tokenize_format_string), batched=True)
    tokenized_datasets.set_format(type='torch')
    print('training sample input',
          tokenizer.decode(pd.DataFrame(tokenized_datasets['train']).iloc[0]['input_ids'], skip_special_tokens=False))
    try:
        print('training sample target',
              tokenizer.decode(pd.DataFrame(tokenized_datasets['train']).iloc[0]['label'], skip_special_tokens=False))
    except Exception as e:
        assert args.architecture == 'causal-lm'
        pass

    ppo_trainer = PPOTrainer(
        config, model, ref_model, tokenizer, train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], data_collator=collator, save_total_limit=args.save_total_limit)
    # device = ppo_trainer.accelerator.device
    # if ppo_trainer.accelerator.num_processes == 1:
    #     device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    # prepare preference_pipe
    preference_model_args = argparse.ArgumentParser()
    preference_model_args_dict = vars(preference_model_args)
    with open(os.path.join(args.preference_model_checkpoint, "args.json"), "r") as f:
        summary_dict = json.load(f)
        preference_model_args_dict.update(summary_dict)
    kwargs = {
        "max_input_length": 512,
        "max_decoder_input_length": 512,
        "only_encode": False,
    }
    reward_pipe = preference_pipe(preference_model_args, args, **kwargs)

    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")

    def compute_metrics(predictions, references):
        x = random.choice(range(len(predictions)))
        print("preds: ", predictions[x])
        print("label: ", references[x])
        predictions, references = postprocess_text(predictions, references)
        print("process_preds: ", predictions[x])
        print("process_label: ", references[x])
        my_metric = final_compute_metrics(predictions, references, metric, metric2, None)
        return my_metric

    generation_kwargs = {
        "do_sample": True,
        "max_length": args.maxlen,
        "temperature": args.temperature,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "eos_token_id": tokenizer.eos_token_id,
        "length_penalty": 2.0,
    }
    global_step = 0
    for epoch in range(args.epochs):
        for idx, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
            global_step += 1
            query_tensors = batch["input_ids"]
            query_tensors = [t.to(ppo_trainer.accelerator.device) for t in query_tensors]
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
            reward_zero = [0 if t[-1] != tokenizer.eos_token_id or len(t) < 6 else 1 for t in response_tensors]
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            texts = [{"QA": batch["QA"][i], "rot": batch["response"][i]} for i in range(len(batch["input_ids"]))]
            # rewards = reward_pipe.compute_score(texts)
            rewards = list(map(lambda x, y: x * y, reward_pipe.compute_score(texts), reward_zero))
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards, labels=batch["label"])
            ppo_trainer.log_stats(stats, batch, rewards)
            if global_step % args.checking_steps == 0:
                ppo_trainer.evaluate(
                    tokenizer, compute_metrics, steps=global_step, output_dir=OUT_DIR, **generation_kwargs)
    ppo_trainer._save_best_model()


if __name__ == '__main__':
    main()
