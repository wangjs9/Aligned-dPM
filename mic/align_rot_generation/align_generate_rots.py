from rot_generation.common import *
from align_rot_generation.models.aligned_bart import AlignedBartForConditionalGeneration
from align_rot_generation.models.aligned_gpt2 import AlignedGPT2LMHeadModel
from align_rot_generation.models.aligned_t5 import AlignedT5ForConditionalGeneration

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from preference_modeling.inputters import inputters
from preference_modeling.utils.building_utils import build_model, deploy_model
import random
import warnings
import re
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")


class Seq2SeqTrainerLogger(Seq2SeqTrainer):
    def __init__(self, logfile, *args, **kwargs):
        super(Seq2SeqTrainer, self).__init__(*args, **kwargs)
        self.logfile = logfile

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.

        Add loss computation about the model updating.
        """
        if self.label_smoother != None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels != None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch != None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        with open(self.logfile, 'a') as outfile:
            outfile.write(json.dumps(output) + '\n')

        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


class AlignedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.

        Add loss computation about the model updating.
        """
        if self.label_smoother != None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels != None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def filter_rot(RoTsList):
    process = lambda x: re.sub("\.(e| ed)$", "", re.sub("[\u00bb»]", "", x)).strip(").'")
    new_RoTsList = []
    for rot_lst in RoTsList:
        rot_lst = [process(rot) + "." for rot in rot_lst]
        new_lst = []
        for rot in rot_lst:
            if rot not in new_lst:
                new_lst.append(rot)
        new_RoTsList.append(new_lst)
    return new_RoTsList


def ranking(toker, model, sample_df, args=None):
    model.eval()
    inputter = inputters["mic"]()

    dataloader_kwargs = {
        "max_input_length": 512,
        "max_decoder_input_length": 512,
        "only_encode": False,
    }

    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos != None, "either eos_token_id or sep_token_id should be provided"

    infer_dataloader = inputter.infer_dataloader(
        toker=toker,
        corpus_df=sample_df,
        batch_size=16,
        **dataloader_kwargs,
    )

    predictions, candidates = [], []
    for batch, sample_idx in infer_dataloader:
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        scores = model.predict(**batch)[:, -1].cpu().numpy()
        predictions.append(scores)
        gold = sample_df.iloc[sample_idx]["rot"]
        all_samples = sample_df.iloc[sample_idx]["rot_samples"]
        cands = {r: s for i, (r, s) in enumerate(zip(all_samples, scores[1:])) if s not in scores[1:i + 1]}
        cands = [gold] + [x[0] for x in sorted(cands.items(), key=lambda x: x[1])]
        candidates.append(cands)

    return predictions, candidates


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


def format_str_to_savefile_name(format_str):
    return '_'.join(
        [x.replace('<', '').replace('>', '') for x in format_str.replace("~", 'TARGET').split() if '[' not in x])


def sample_rot_causal_lm(args, df, model, tokenizer, skip_special_tokens=True, remove_history=False):
    if os.path.exists(os.path.join(args.input, f'train_sample_{args.sample_num}.json')):
        with open(os.path.join(args.input, f'train_sample_{args.sample_num}.json'), 'r') as f:
            train_sample = f.readlines()
            train_sample = [json.loads(x) for x in train_sample]
        return train_sample
    else:
        model = model.to('cuda')
        eos_token_id = tokenizer.encode("<eos>", add_special_tokens=False)[0]
        generations = []
        assert args.sample_num > 1
        for _, row in tqdm(df.iterrows(), total=len(df)):
            input_ids = torch.tensor([row['input_ids']], device='cuda')

            out = model.generate(
                input_ids,
                do_sample=False,
                max_length=args.maxlen,
                num_beams=args.sample_num,
                num_beam_groups=args.sample_num,
                num_return_sequences=args.sample_num,
                early_stopping=True,
                pad_token_id=50256,
                no_repeat_ngram_size=3,
                eos_token_id=eos_token_id,
                diversity_penalty=4.0,
                length_penalty=-1.0,
            )
            if remove_history:
                generations.append(
                    tokenizer.batch_decode(out[:, input_ids.shape[-1]:], skip_special_tokens=skip_special_tokens))

            else:
                generations.append(tokenizer.batch_decode(out, skip_special_tokens=skip_special_tokens))
        with open(os.path.join(args.input, f'train_sample_{args.sample_num}.json'), 'w') as f:
            for line in generations:
                f.write(json.dumps(line) + '\n')
        return filter_rot(generations)


def sample_rot_seq2seq(args, tokenized_dataset, model, tokenizer):
    if os.path.exists(os.path.join(args.input, f'train_sample_{args.sample_num}.json')):
        with open(os.path.join(args.input, f'train_sample_{args.sample_num}.json'), 'r') as f:
            train_sample = f.readlines()
            train_sample = [json.loads(x) for x in train_sample]
        return train_sample
    else:
        model = model.to('cuda')
        assert args.sample_num > 1

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.input,
            per_device_eval_batch_size=args.batchsize,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            predict_with_generate=True,
            fp16=False if "flan-t5" in args.model_checkpoint else True,
            seed=args.seed,
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
        trainer = SampleSeq2SeqTrainer(
            model,
            training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        raw_pred, _, _ = trainer.predict(tokenized_dataset, num_beams=args.sample_num)
        generations = tokenizer.batch_decode(raw_pred, skip_special_tokens=True)
        generations = np.array(generations).reshape(-1, args.sample_num)
        generations = generations.tolist()

        with open(os.path.join(args.input, f'train_sample_{args.sample_num}.json'), 'w') as f:
            for line in generations:
                f.write(json.dumps(line) + '\n')

        return filter_rot(generations)


def generate_aligned_dataset(args, model, tokenizer):
    train_dataset = load_dataset("csv", data_files=os.path.join(args.input, 'tmp/train.csv'))["train"]
    print('train size:', len(train_dataset))

    # add special tokens to tokenizer
    special_tokens = list(
        set(
            get_all_attributes(pd.DataFrame(train_dataset))
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
    model.resize_token_embeddings(len(tokenizer))

    print("Do diverse beam search ...")
    tokenize_format_string = args.format_string.replace("~",
                                                        "") if args.architecture == 'causal-lm' else args.format_string
    tokenized_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer, tokenize_format_string), batched=True)
    sample_df = pd.DataFrame(tokenized_dataset)[[args.source_name, args.target_name]]
    if args.architecture == 'causal-lm':
        sample_df[args.target_name + '_samples'] = sample_rot_causal_lm(
            args, sample_df, model, tokenizer, skip_special_tokens=True, remove_history=False)
    else:
        sample_df[args.target_name + '_samples'] = sample_rot_seq2seq(args, tokenized_dataset, model, tokenizer)
    # load preference_model
    mode = "d-PM" if "d-PM" in args.preference_model_checkpoint else "soft"
    preference_model_args = argparse.ArgumentParser()
    annotator_args_dict = vars(preference_model_args)
    with open(os.path.join(args.preference_model_checkpoint, "args.json"), "r") as f:
        summary_dict = json.load(f)
        annotator_args_dict.update(summary_dict)
    kwargs = {'mode': mode}
    preference_toker, preference_model = build_model(
        checkpoint=os.path.join(args.preference_model_checkpoint, "best.bin"),
        args=preference_model_args, **kwargs,
    )
    preference_model_args.n_gpu = 1
    preference_model_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    preference_model = deploy_model(preference_model, preference_model_args)
    scores, candidates = ranking(preference_toker, preference_model, sample_df, args)
    sample_df[args.target_name + '_scores'], sample_df[args.target_name + '_candidates'] = scores, candidates
    sample_df.to_json(
        os.path.join(args.input, f'train_candidates_{mode}_{args.sample_num}.json'), orient="records", indent=4)
    train_dataset = train_dataset.add_column(
        name=args.target_name + '_candidates', column=sample_df[args.target_name + '_candidates'])
    pd.DataFrame(train_dataset).to_json(
        os.path.join(args.input, f'tmp/aligned_{mode}_{args.sample_num}.json'), orient="records", indent=4)


import nltk


def clac_metric(decoder_preds, decoder_labels, no_glove=False):
    ref_list = []
    hyp_list = []
    for ref, hyp in zip(decoder_labels, decoder_preds):
        ref = ' '.join(nltk.word_tokenize(ref.lower()))
        hyp = ' '.join(nltk.word_tokenize(hyp.lower()))
        if len(hyp) == 0:
            hyp = '&'
        ref_list.append(ref)
        hyp_list.append(hyp)

    from metric import NLGEval
    metric = NLGEval(no_glove=no_glove)
    metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
    return metric_res


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


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
    parser.add_argument('--maxlen', type=int, default=512,
                        help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--warmup_steps', type=int, default=8000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
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

    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--preference_model_checkpoint', type=str,
                        default="../preference_modeling/output/mic_d-PM_23-0528-1653")
    parser.add_argument('--checking_steps', type=int, default=2000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" if "bart-large" in args.model_checkpoint else str(args.gpu)

    set_seed(args.seed)

    # configure for architecture
    if args.architecture == '':
        if 'gpt' in args.model_checkpoint:
            args.architecture = 'causal-lm'
            AutoModel = AlignedGPT2LMHeadModel
        elif 'bart' in args.model_checkpoint:
            args.architecture = 'seq2seq'
            AutoModel = AlignedBartForConditionalGeneration
        elif 't5' in args.model_checkpoint:
            args.architecture = 'seq2seq'
            AutoModel = AlignedT5ForConditionalGeneration
    else:
        AutoModel = AutoModelForCausalLM if (args.architecture == 'causal-lm') else AutoModelForSeq2SeqLM

    assert os.path.exists(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModel.from_pretrained(args.model_checkpoint)

    # prepare out directory
    mode = "d-PM" if "d-PM" in args.preference_model_checkpoint else "soft"

    OUT_DIR = os.path.join(
        args.output,
        f"{args.model_checkpoint.split('/')[-2].split('_')[0]}_{format_str_to_savefile_name(args.format_string)}_sample{args.sample_num}_epochs{args.epochs}_batch{args.batchsize}_seed{args.seed}_{mode}")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")

    if not os.path.exists(os.path.join(args.input, f'tmp/aligned_{mode}_{args.sample_num}.json')):
        generate_aligned_dataset(args, model, tokenizer)

    aligned_dataset = load_dataset(
        "json", data_files=os.path.join(args.input, f'tmp/aligned_{mode}_{args.sample_num}.json'))
    dataset = load_dataset("csv", data_files={'test': os.path.join(args.input, 'tmp/test.csv'),
                                              'validation': os.path.join(args.input, 'tmp/dev.csv')})

    print('train size:', len(aligned_dataset['train']))

    # add special tokens to tokenizer
    special_tokens = list(
        set(
            get_all_attributes(pd.DataFrame(aligned_dataset['train']))
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
    model.resize_token_embeddings(len(tokenizer))
    init_attribute_embeddings(model, tokenizer, special_tokens)

    args.pad_token_id = tokenizer.pad_token_id

    # tokenize_format_string = args.format_string.replace("~",
    #                                                     "") if args.architecture == 'causal-lm' else args.format_string
    tokenize_format_string = args.format_string
    tokenized_datasets = dataset.map(
        lambda x: preprocess(x, tokenizer, tokenize_format_string),
        batched=True,
        load_from_cache_file=False
    )

    max_sample_num = max([len(x) for x in aligned_dataset['train']['rot_candidates']])
    print(f"max_sample_num: {max_sample_num}")
    aligned_tokenize_format_string = tokenize_format_string.replace(" rot", " rot_candidates")
    aligned_tokenized_datasets = aligned_dataset.map(
        lambda x: aligned_preprocess(x, tokenizer, aligned_tokenize_format_string, max_sample_num),
        batched=True,
        load_from_cache_file=False
    )
    print('training sample input', tokenizer.decode(
        pd.DataFrame(aligned_tokenized_datasets['train']).iloc[0]['input_ids'], skip_special_tokens=False))
    try:
        print('training sample target', tokenizer.decode(
            pd.DataFrame(aligned_tokenized_datasets['train']).iloc[0]['labels'], skip_special_tokens=False))
    except Exception as e:
        # it doesn't matter; this means there is no "labels" in the dataset because we are using causal-lm
        pass

    data_collator = AlignedDataCollatorForLanguageModeling(tokenizer, mlm=False) if (args.architecture == 'causal-lm') \
        else AlignedDataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # print()
        if isinstance(preds, tuple):
            preds = preds[0]
        # print("one: before decoder")
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        x = random.choice(range(len(decoded_labels)))
        print("preds: ", decoded_preds[x])
        print("label: ", decoded_labels[x])
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        print("process_preds: ", decoded_preds[x])
        print("process_label: ", decoded_labels[x])
        # my_metric = clac_metric(decoder_preds=decoded_preds, decoder_labels=decoded_labels)
        my_metric = final_compute_metrics(decoded_preds, decoded_labels, metric, metric2, None)
        return my_metric

    results = {}
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    training_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "checkpoints"),
        evaluation_strategy="steps",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batchsize,
        per_device_eval_batch_size=20,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        fp16=False if "flan-t5" in args.model_checkpoint else True,
        seed=args.seed,
        save_strategy='steps',
        logging_steps=args.checking_steps,
        save_steps=args.checking_steps,
        eval_steps=args.checking_steps,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        load_best_model_at_end=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    trainer = AlignedTrainer(
        model,
        training_args,
        train_dataset=aligned_tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    if args.architecture == 'seq2seq':
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(OUT_DIR, "checkpoints"),
            evaluation_strategy="steps",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batchsize,
            per_device_eval_batch_size=20,
            weight_decay=args.weight_decay,
            save_total_limit=args.save_total_limit,
            num_train_epochs=args.epochs,
            predict_with_generate=True,
            fp16=False if "flan-t5" in args.model_checkpoint else True,
            seed=args.seed,
            save_strategy='steps',
            logging_steps=args.checking_steps,
            save_steps=args.checking_steps,
            eval_steps=args.checking_steps,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
            load_best_model_at_end=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_num_workers=args.dataloader_num_workers,
        )

        trainer = Seq2SeqTrainerLogger(
            os.path.join(OUT_DIR, 'log.txt'),
            model,
            training_args,
            train_dataset=aligned_tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler),
        )
    # trainer.evaluate(tokenized_datasets["validation"])
    trainer.train()

    trainer.save_model(os.path.join(OUT_DIR, "model"))

    # tokenize again if using causal-lm because before, the test set did not include targets
    set_seed(args.seed)
    if args.architecture == 'causal-lm':
        tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer, args.format_string), batched=True)

    out_df = pd.DataFrame(tokenized_datasets['test'])[[args.source_name, args.target_name, 'input_ids']]

    # results_df = pd.DataFrame().from_dict(results, orient='index')
    # print(results_df)
    # results_df.to_csv(os.path.join(OUT_DIR, f'results_epochs{args.epochs}_batch{args.batchsize}_seed{args.seed}.csv'))

    out_df[args.target_name + '_generated'] = decode(
        args,
        out_df,
        trainer.model,
        tokenizer,
        remove_history=(args.architecture == 'causal-lm'),
        skip_special_tokens=True
    )
    out_df.to_csv(os.path.join(
        OUT_DIR, f'test_generations_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}.csv'))

    with open(os.path.join(OUT_DIR, 'format_string.txt'), 'w') as outfile:
        outfile.write(args.format_string)

    results = final_compute_metrics(
        out_df[args.target_name + '_generated'].values, out_df[args.target_name].values, metric, metric2, tokenizer)

    fn = os.path.join(OUT_DIR, f'results_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}.json')
    with open(fn, 'w') as outfile:
        json.dump(results, outfile)

    torch.save(args, os.path.join(OUT_DIR, "training_args.bin"))


if __name__ == '__main__':
    main()
