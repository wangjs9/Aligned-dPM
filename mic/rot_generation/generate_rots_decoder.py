from rot_generation.common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', type=str)
    parser.add_argument('--input', type=str, help='path to input file')
    parser.add_argument('--output', type=str, default='results', help='path to directory for outputting results')
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    parser.add_argument('--gpu', type=int, default=1, choices=list(range(8)))
    parser.add_argument('--source_name', type=str, default='QA')
    parser.add_argument('--target_name', type=str, default='rot')
    parser.add_argument('--format_string', type=str, default="Q [answ] A [rot] ~ rot", help='how to format the dataset')
    parser.add_argument('--maxlen', type=int, default=512,
                        help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded')
    parser.add_argument('--beams', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--skip_special_tokens', action='store_true')

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)

    # prepare out directory
    OUT_DIR = args.output
    if not os.path.exists(os.path.join(OUT_DIR, 'tmp')):
        os.makedirs(os.path.join(OUT_DIR, 'tmp'))

    df = pd.read_csv(args.input)
    df[df['split'] == 'train'].to_csv(os.path.join(OUT_DIR, 'tmp/train.csv'), index=False)
    df[df['split'] == 'test'].to_csv(os.path.join(OUT_DIR, 'tmp/test.csv'), index=False)

    dataset = load_dataset('csv', data_files={
        'train': os.path.join(OUT_DIR, 'tmp/train.csv'),
        'test': os.path.join(OUT_DIR, 'tmp/test.csv')
    })
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory)

    AutoModel = AutoModelForCausalLM if 'gpt' in args.model_directory else AutoModelForSeq2SeqLM

    model = AutoModel.from_pretrained(args.model_directory).to('cuda')

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

    tokenized_datasets = dataset.map(lambda x: preprocess(x, tokenizer, args.format_string), batched=True)

    out_df = pd.DataFrame(tokenized_datasets['test'])[[args.source_name, args.target_name, 'input_ids']]

    out_df[args.target_name + '_generated'] = decode(
        args,
        out_df,
        model,
        tokenizer,
        skip_special_tokens=args.skip_special_tokens,
        remove_history=('gpt' in args.model_directory)
    )
    out_df.to_csv(os.path.join(args.output,
                               f'test_generations_beams{args.beams}_p{args.top_p}_k{args.top_k}_temp{args.temperature}_seed{args.seed}.csv'))


if __name__ == '__main__':
    main()
