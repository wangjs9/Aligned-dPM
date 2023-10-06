from rot_generation.common import *


class Seq2SeqTrainerLogger(Seq2SeqTrainer):
    def __init__(self, logfile, *args, **kwargs):
        super(Seq2SeqTrainer, self).__init__(*args, **kwargs)
        self.logfile = logfile

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


class TrainerLogger(Trainer):
    def __init__(self, logfile, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.logfile = logfile

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
    result['sacrebleu'] = round(result2["score"], 1)

    r = {k: round(v, 4) for k, v in result.items()}

    return r


def format_str_to_savefile_name(format_str):
    return '_'.join(
        [x.replace('<', '').replace('>', '') for x in format_str.replace("~", 'TARGET').split() if '[' not in x])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='t5-small',
                        choices=['gpt2', 't5-small', 'google/flan-t5-base', 'facebook/bart-large'])
    parser.add_argument('--architecture', default='', choices=['seq2seq', 'causal-lm'])
    parser.add_argument('--input', type=str, help='path to input file (MIC dataset)')
    parser.add_argument('--output', type=str, default='results', help='path to directory for outputting results')
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--format_string', type=str, default="Q [answ] A [rot] ~ rot", help='how to format the dataset')
    parser.add_argument('--source_name', type=str, default='QA',
                        help='the name of the source column to write in the out file')
    parser.add_argument('--target_name', type=str, default='rot',
                        help='the name of the target column to write in the out file')
    parser.add_argument('--maxlen', type=int, default=512,
                        help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--generate_attributes', action='store_true')
    parser.add_argument('--train_size', type=int, default=-1,
                        help='the number of train datapoints to use as an ablation (default is to use the full train set)')

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)

    # configure for architecture
    if args.architecture == '':
        if 'gpt' in args.model_checkpoint:
            args.architecture = 'causal-lm'
        elif ('bart' in args.model_checkpoint) or ('t5' in args.model_checkpoint):
            args.architecture = 'seq2seq'
    AutoModel = AutoModelForCausalLM if (args.architecture == 'causal-lm') else AutoModelForSeq2SeqLM

    # prepare out directory
    OUT_DIR = os.path.join(args.output,
                           f"{args.model_checkpoint.split('/')[-1]}_{format_str_to_savefile_name(args.format_string)}_epochs{args.epochs}_batch{args.batchsize}_seed{args.seed}")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if not os.path.exists(os.path.join(OUT_DIR, 'tmp')):
        os.makedirs(os.path.join(OUT_DIR, 'tmp'))

    df = pd.read_csv(args.input)  # 'clean_moral_eval_jun_25.csv'
    for s in ['train', 'dev', 'test']:
        if (s == 'train') and (args.train_size > 0):
            df[df['split'] == s].sample_rot(n=args.train_size, random_state=args.seed).to_csv(
                os.path.join(OUT_DIR, 'tmp/%s.csv' % s), index=False)
        else:
            df[df['split'] == s].to_csv(os.path.join(OUT_DIR, 'tmp/%s.csv' % s), index=False)
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")

    dataset = load_dataset('csv', data_files={'train': os.path.join(OUT_DIR, 'tmp/train.csv'),
                                              'test': os.path.join(OUT_DIR, 'tmp/test.csv'),
                                              'validation': os.path.join(OUT_DIR, 'tmp/dev.csv')})
    print('train size:', len(dataset['train']))
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    model = AutoModel.from_pretrained(args.model_checkpoint)

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
    model.resize_token_embeddings(len(tokenizer))
    init_attribute_embeddings(model, tokenizer, special_tokens)

    args.pad_token_id = tokenizer.pad_token_id

    tokenize_format_string = args.format_string.replace("~",
                                                        "") if args.architecture == 'causal-lm' else args.format_string
    tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer, tokenize_format_string), batched=True)

    print('training sample input',
          tokenizer.decode(pd.DataFrame(tokenized_datasets['train']).iloc[0]['input_ids'], skip_special_tokens=False))
    try:
        print('training sample target',
              tokenizer.decode(pd.DataFrame(tokenized_datasets['train']).iloc[0]['labels'], skip_special_tokens=False))
    except Exception as e:
        # it doesn't matter; this means there is no "labels" in the dataset because we are using causal-lm
        pass

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) if (
            args.architecture == 'causal-lm') else DataCollatorForSeq2Seq(tokenizer, model=model)

    results = {}

    training_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "checkpoints"),
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batchsize,
        per_device_eval_batch_size=args.batchsize,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        fp16=False if args.model_checkpoint == "google/flan-t5-base" else True,
        seed=args.seed,
        save_strategy='epoch'
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    if args.architecture == 'seq2seq':
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(OUT_DIR, "checkpoints"),
            evaluation_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batchsize,
            per_device_eval_batch_size=args.batchsize,
            weight_decay=args.weight_decay,
            save_total_limit=args.save_total_limit,
            num_train_epochs=args.epochs,
            predict_with_generate=True,
            fp16=False if args.model_checkpoint == "google/flan-t5-base" else True,
            seed=args.seed,
            save_strategy='epoch'
        )
        trainer = Seq2SeqTrainerLogger(
            os.path.join(OUT_DIR, 'log.txt'),
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )

    trainer.train()

    trainer.save_model(os.path.join(OUT_DIR, "model"))

    # tokenize again if using causal-lm because before, the test set did not include targets
    if args.architecture == 'causal-lm':
        tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer, args.format_string), batched=True)

    out_df = pd.DataFrame(tokenized_datasets['test'])[[args.source_name, args.target_name, 'input_ids']]

    results_df = pd.DataFrame().from_dict(results, orient='index')
    print(results_df)
    results_df.to_csv(os.path.join(OUT_DIR,
                                   f'results_epochs{args.epochs}_batch{args.batchsize}_lr{int(args.lr * (10 ** 5))}_seed{args.seed}.csv'))

    if (args.top_p <= 0) and (args.top_k <= 0) and (args.beams <= 0) and (args.architecture == 'seq2seq'):
        raw_pred, _, _ = trainer.predict(tokenized_datasets['test'])
        out_df[args.target_name + '_generated'] = tokenizer.batch_decode(raw_pred, skip_special_tokens=True)
    else:
        out_df[args.target_name + '_generated'] = decode(args,
                                                         out_df,
                                                         trainer.model,
                                                         tokenizer,
                                                         remove_history=(args.architecture == 'causal-lm'),
                                                         skip_special_tokens=True
                                                         )
    out_df.to_csv(os.path.join(OUT_DIR,
                               f'test_generations_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}.csv'))

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
