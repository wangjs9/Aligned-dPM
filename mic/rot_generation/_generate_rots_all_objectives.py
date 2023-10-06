from generate_rots import *


def df_to_source_target(df, args):
    ## should use the [rot_and_attr] tag
    sc_format_strings = [
        '[situation] situation [rot] ~ rot',  # text-only generation
        '[situation] situation [rot] rot [situation_attr] ~ <moral> <rot-agree> <violation-severity>',
        # attribute prediction
        '[situation] situation [situation_attr] <moral> <rot-agree> <violation-severity> ~ [rot] rot',
        # controlled generation
        '[situation] situation [rot] rot [situation_attr] ~ <moral> <rot-agree> <violation-severity>',
        # attribute labeling
        '[situation] situation [situation_attr] ~ <moral> <rot-agree> <violation-severity> [rot] rot',
        # model choice generation
    ]

    qa_format_strings = [
        'Q [answ] A [rot] ~ rot',  # text-only generation
        'Q [answ] A [rot] rot [attr] ~ <A_agrees> <moral> <rot-agree> <violation-severity>',  # attribute prediction
        'Q [answ] A [attr] <A_agrees> <moral> <rot-agree> <violation-severity> ~ [rot] rot',  # controlled generation
        'Q [answ] A [rot] rot [attr] ~ <A_agrees> <moral> <rot-agree> <violation-severity>',  # attribute labeling
        'Q [answ] A [attr] ~ <A_agrees> <moral> <rot-agree> <violation-severity> [rot] rot',  # model choice generation
    ]

    format_strings = lambda row: qa_format_strings if 'Q' in row and type(row['Q']) == str else sc_format_strings
    proc = lambda format_string: format_string.replace("~", "") if args.architecture == 'causal-lm' else format_string

    source_target = [build(row, proc(format_string))
                     for _, row in df.iterrows() for format_string in format_strings(row)]
    source = [tup[0] for tup in source_target]
    target = [tup[1] if len(tup) > 1 else "" for tup in source_target]

    return pd.DataFrame({'source': source, 'target': target})


def preprocess_source_target(examples, tokenizer):
    source = examples['source']
    target = [x if x else "" for x in examples['target']]

    model_inputs = tokenizer(source)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='t5-small')
    parser.add_argument('--architecture', default='', choices=['seq2seq', 'causal-lm'])
    parser.add_argument('--input', type=str, help='path to input file')
    parser.add_argument('--output', type=str, default='results', help='path to directory for outputting results')
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--maxlen', type=int, default=512,
                        help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--generate_attributes', action='store_true')
    parser.add_argument('--train_size', type=int, default=-1)

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
                           f"{args.model_checkpoint.split('/')[-1]}_all_objectives_epochs{args.epochs}_batch{args.batchsize}")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if not os.path.exists(os.path.join(OUT_DIR, 'tmp')):
        os.makedirs(os.path.join(OUT_DIR, 'tmp'))

    df = pd.read_csv(args.input)  # 'clean_moral_eval_jun_25.csv'
    for s in ['train', 'dev', 'test']:
        shuffle = lambda df: df if s == 'test' else df.sample_rot(frac=1, random_state=args.seed)
        if (s == 'train') and (args.train_size > 0):
            shuffle(df_to_source_target(df[df['split'] == s].copy(), args)).sample_rot(n=args.train_size,
                                                                                       random_state=args.seed).to_csv(
                os.path.join(OUT_DIR, 'tmp/%s.csv' % s), index=False)
        else:
            shuffle(df_to_source_target(df[df['split'] == s].copy(), args)).to_csv(
                os.path.join(OUT_DIR, 'tmp/%s.csv' % s), index=False)
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
                "[situation]",
                "[situation_attr]",
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

    tokenized_datasets = dataset.map(lambda x: preprocess_source_target(x, tokenizer), batched=True)

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
        fp16=True,
        seed=args.seed
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
            fp16=True,
            seed=args.seed
        )
        trainer = Seq2SeqTrainerLogger(
            os.path.join(OUT_DIR, 'log.txt'),
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda x: compute_metrics(x, metric, metric2, tokenizer, dictionary=results),
        )

    trainer.train()

    trainer.save_model(os.path.join(OUT_DIR, "model"))

    # tokenize again if using causal-lm because before, the test set did not include targets
    if args.architecture == 'causal-lm':
        tokenized_datasets = dataset.map(lambda x: preprocess_source_target(x, tokenizer), batched=True)

    out_df = pd.DataFrame(tokenized_datasets['test'])[['source', 'target', 'input_ids']]

    results_df = pd.DataFrame().from_dict(results, orient='index')
    print(results_df)
    results_df.to_csv(os.path.join(OUT_DIR,
                                   f'results_epochs{args.epochs}_batch{args.batchsize}_lr{int(args.lr * (10 ** 5))}_seed{args.seed}.csv'))

    if (args.top_p <= 0) and (args.top_k <= 0) and (args.beams <= 0) and (args.architecture == 'seq2seq'):
        raw_pred, _, _ = trainer.predict(tokenized_datasets['test'])
        out_df['target_generated'] = tokenizer.batch_decode(raw_pred, skip_special_tokens=True)
    else:
        out_df['target_generated'] = decode(args,
                                            out_df,
                                            trainer.model,
                                            tokenizer,
                                            remove_history=(args.architecture == 'causal-lm'),
                                            skip_special_tokens=True
                                            )
    out_df.to_csv(os.path.join(OUT_DIR,
                               f'test_generations_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}.csv'))

    results = final_compute_metrics(out_df['target_generated'].values,
                                    out_df['target'].values,
                                    metric, metric2, tokenizer)

    fn = os.path.join(args.output, f'results_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}.json')
    with open(fn, 'w') as outfile:
        json.dump(results, outfile)

    torch.save(args, os.path.join(OUT_DIR, "training_args.bin"))


if __name__ == '__main__':
    main()