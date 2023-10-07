import os
import shutil
import sys
import argparse
import time
import datetime
import json
from tqdm import tqdm
import logging
import wandb

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributed import get_rank
from transformers.optimization import AdamW
# from torch.optim import Adam
from transformers.trainer_utils import set_seed

from inputters import inputters
from utils.building_utils import boolean_string, build_model, deploy_model
from utils.align_utils import AlignmentDataLoader, RankingLoss, candidate_sampling, test_model, eval_model
import warnings

warnings.filterwarnings("ignore")


def calibrate_one_batch(base_model, toker, input_ids=None, attention_mask=None, candidate_ids=None, normalize=True,
                        score_mode="base", length_penalty=1, require_gold=True, adding=1e-8, **kwargs):
    batch_size = input_ids.size(0)
    candidate_num = candidate_ids.size(1)
    candidate_mask = candidate_ids != toker.pad_token_id

    encoder_outputs = base_model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    encoder_hidden_states = encoder_outputs[0]
    encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, candidate_num, dim=0)
    encoder_outputs.last_hidden_state = encoder_hidden_states

    decoder_input_ids = candidate_ids.view(-1, candidate_ids.size(-1))
    decoder_attention_mask = candidate_mask.view(-1, candidate_mask.size(-1))
    output = base_model(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask, align=True)

    output = output[0]  # [bz x cand_num, seq_len, word_dim]
    output = output.view(batch_size, -1, output.size(1), output.size(2))  # [bz, cand_num, seq_len, word_dim]
    output = output[:, :, :-1]  # truncate last token
    probs = output[:, 0].contiguous()

    # add eos token
    candidate_len = torch.sum(candidate_ids.ne(toker.pad_token_id), dim=2, keepdim=True).type_as(candidate_ids)
    eos_position = torch.remainder(candidate_len, candidate_ids.size(2))
    candidate_ids = candidate_ids.scatter(2, eos_position, toker.eos_token_id)
    candidate_ids = candidate_ids[:, :, 1:]  # shift right --> labels
    gold_ids = candidate_ids[:, 0].contiguous()
    # get candidate mask
    candidate_mask = candidate_ids != toker.pad_token_id
    candidate_ids = candidate_ids.unsqueeze(-1)

    # compute lm_loss
    masked_lm_loss = F.cross_entropy(probs.view(-1, probs.size(-1)), gold_ids.view(-1), ignore_index=toker.pad_token_id)
    ppl_value = torch.exp(masked_lm_loss)

    # compute scores
    if normalize:
        if score_mode == "log":
            _output = F.log_softmax(output, dim=3)
        else:
            _output = F.softmax(output, dim=3)
        scores = torch.gather(_output, 3, candidate_ids).squeeze(-1)  # [bz, cand_num, seq_len]
    else:
        scores = torch.gather(output, 3, candidate_ids).squeeze(-1)  # [bz, cand_num, seq_len]
    candidate_mask = candidate_mask.float()
    scores = torch.mul(scores, candidate_mask).sum(-1) / (
            (candidate_mask.sum(-1) + adding) ** length_penalty)  # [bz, cand_num]
    if require_gold:
        output = {"cand_score": scores[:, 1:], "gold_score": scores[:, 0], "lm_loss": masked_lm_loss, "ppl": ppl_value}
    else:
        output = {"cand_score": scores, "probs": probs}
    return output


def align_base_model(base_model, toker, train_dataloader, args, output_dir, logger=None):
    INF = 100000000
    CACHE_EMPTY_STEP = 10000
    set_seed(args.align_seed)
    step_cnt = 0
    all_step_cnt = 0
    epoch = 0
    # lowest_eval_loss = 20
    highest_score = {}

    if args.local_rank == -1 or get_rank() == 0:
        log_output_dir = os.path.join(output_dir, "log_files")
        if not os.path.exists(log_output_dir):
            os.mkdir(log_output_dir)
        train_logger = open(os.path.join(log_output_dir, "alignment_train_log.csv"), "a+", buffering=1)
        eval_logger = open(os.path.join(log_output_dir, "alignment_eval_log.csv"), "a+", buffering=1)
        print("all_step_cnt\tstep_cnt\tavg_loss\tavg_ranking_loss\tavg_lm_loss\tavg_ppl\tn_token_total\tepoch_time",
              file=train_logger)
        print("all_step_cnt\tstep_cnt\teval_loss\teval_ppl\teval_bleu\teval_rouge", file=eval_logger)

    optim_step_num = len(train_dataloader) // (args.train_batch_size * args.accumulate_step) + int(
        len(train_dataloader) % (args.train_batch_size * args.accumulate_step) != 0)
    epoch_num = args.epoch_num
    valid_step = args.valid_step * args.accumulate_step
    if epoch_num != None:
        optim_step_num = optim_step_num * epoch_num

    if args.local_rank != -1:
        args.n_gpu = 1
    if args.local_rank == -1 or get_rank() == 0:
        if args.pbar:
            pbar = tqdm(total=optim_step_num, desc=f"Aligning base model *{args.config_name}* with *{args.mode}*")
        else:
            pbar = None
    else:
        pbar = None

    inputter = inputters[args.inputter_name]()

    param_optimizer = list(base_model.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']  # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    # monitor the loss of model
    wandb.init(config=args, project="aligned-ESConv", entity="align_d-pm")
    last_update_time = 0
    while True:
        base_model.train()
        (avg_ranking_loss, avg_lm_loss, avg_loss, avg_ppl) = 0.0, 0.0, 0.0, 0.0
        train_start_time_epoch = time.time()
        for batch in train_dataloader:
            # activate new training mode
            batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            batch.update({"all_step_cnt": all_step_cnt})
            batch.update({"epoch": epoch})
            batch.update({"warmup_steps": args.warmup_step})
            outputs = calibrate_one_batch(base_model, toker, **batch)

            # obtain generation loss and ppl
            lm_loss = outputs.pop("lm_loss")
            ppl = outputs.pop("ppl")
            # compute ranking loss
            cand_similarity = outputs.pop("cand_score") * args.scale
            gold_similarity = outputs.pop("gold_score") * args.scale
            if cand_similarity.size(1) == 0:
                ranking_loss = 0
            else:
                ranking_loss = RankingLoss(
                    cand_similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight)

            if args.n_gpu > 1:
                ranking_loss = ranking_loss.mean() if ranking_loss != 0 else 0
                lm_loss = lm_loss.mean()
                ppl = lm_loss.mean()

            loss = (args.rank_weight * ranking_loss + args.lm_weight * lm_loss) / args.accumulate_step

            avg_loss += loss.item()
            avg_lm_loss += lm_loss.item() / args.accumulate_step
            avg_ranking_loss += ranking_loss.item() / args.accumulate_step if ranking_loss != 0 else 0
            if ppl.item() < INF:
                avg_ppl += ppl.item()

            if args.fp16:
                NotImplementedError
            else:
                loss.backward()

            # gradient update
            step_cnt += 1
            if step_cnt % args.accumulate_step == 0:
                # updating
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.max_grad_norm)
                all_step_cnt += 1
                # adjust learning rate
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_step ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()

                # Print log info to file
                if args.local_rank == -1 or get_rank() == 0:
                    avg_op = lambda x: x / args.accumulate_step
                    avg_ranking_loss, avg_lm_loss, avg_loss, avg_ppl = avg_op(avg_ranking_loss), avg_op(
                        avg_lm_loss), avg_op(avg_loss), avg_op(avg_ppl)
                    epoch_time = time.time() - train_start_time_epoch
                    if pbar is not None:
                        pbar_str = ""
                        for k, v in outputs.items():
                            if args.n_gpu > 1:
                                pbar_str += f"{k}: {v.mean().item():.2f} "
                            else:
                                pbar_str += f"{k}: {v.item():.2f} "
                        pbar_str += f"ppl: {avg_ppl:.2f} ranking_loss: {avg_ranking_loss: .3f} lm_loss: {avg_lm_loss: .2f} epoch: {epoch + 1}"

                        pbar.set_postfix_str(pbar_str)
                        pbar.update(1)

                    print(
                        f"{epoch + 1}\t{all_step_cnt + 1}\t{step_cnt + 1}\t{avg_loss}\t{avg_ranking_loss}\t{avg_lm_loss}\t{avg_ppl}\t{epoch_time}",
                        file=train_logger)

                    wandb.log({"avg_ranking_loss": avg_ranking_loss, "avg_lm_loss": avg_lm_loss, "avg_loss": avg_loss})

                    avg_ranking_loss, avg_lm_loss, avg_loss, avg_ppl = 0.0, 0.0, 0.0, 0.0

                if step_cnt % valid_step == 0:
                    if args.local_rank == -1 or get_rank() == 0:
                        set_seed(args.align_seed)
                        eval_dataloader_loss = inputter.valid_dataloader(
                            toker=toker,
                            corpus_file=args.eval_input_file,
                            batch_size=args.eval_batch_size,
                            **args.dataloader_kwargs
                        )
                        eval_dataloader_infer = inputter.infer_dataloader(
                            infer_input_file=args.eval_input_file,
                            toker=toker,
                            **args.infer_dataloader_kwargs
                        )
                        eval_loss, eval_ppl, metric_results, metric_result_list, results = eval_model(
                            base_model, toker, args, eval_dataloader_loss, eval_dataloader_infer
                        )
                        eval_bleu = metric_results["bleu-4"]
                        eval_rouge = metric_results["rouge-l"]
                        print(
                            f"**Eval (step: {all_step_cnt})** Loss: {eval_loss:.4f}, PPL: {eval_ppl:.4f}, BLEU-4: {eval_bleu:.2f}, ROUGE-L: {eval_rouge:.2f}")
                        eval_score = eval_bleu
                        if eval_loss > 5:
                            logger.info("ppl is greater than 5.")
                            exit()
                        last_update_time += 1

                        if len(highest_score) < args.save_total_limit or eval_score > min(highest_score.keys()):
                            last_update_time = 0
                            args.checkpoint_dir = os.path.join(output_dir, f"aligned_{args.config_name}_{all_step_cnt}")
                            highest_score[eval_score] = args.checkpoint_dir
                            os.makedirs(args.checkpoint_dir, exist_ok=True)
                            torch.save(base_model.state_dict(), os.path.join(args.checkpoint_dir, "pytorch_model.bin"))
                            toker.save_vocabulary(args.checkpoint_dir)

                            with open(os.path.join(args.checkpoint_dir, "valid_metrics.json"), "w",
                                      encoding="utf-8") as f:
                                json.dump(metric_results, f, indent=2, ensure_ascii=False, sort_keys=False)

                            if len(highest_score) > args.save_total_limit:
                                shutil.rmtree(highest_score[min(highest_score.keys())])
                                del highest_score[min(highest_score.keys())]

                        print(
                            f"{epoch + 1}\t{all_step_cnt}\t{step_cnt}\t{eval_loss}\t{eval_ppl}\t{eval_bleu}\t{eval_rouge}",
                            file=eval_logger)
                        logger.info("current learning rate: " + str(optimizer.param_groups[0]["lr"]))
                        wandb.log({"eval_loss": eval_loss, "eval_ppl": eval_ppl, "eval_bleu": eval_bleu,
                                   "eval_rouge": eval_rouge})
                        base_model.train()

                if args.epoch_num is None and all_step_cnt >= optim_step_num:
                    logger.info(f"all_step_cnt {all_step_cnt} is greater than optim_step_num {optim_step_num}")
                    break

                if last_update_time > 10:
                    logger.info(f"teh model doesn't increase its performance for 10 steps")
                    break

            if (step_cnt + 1) % CACHE_EMPTY_STEP == 0:
                torch.cuda.empty_cache()

        if args.epoch_num is not None:
            epoch += 1

            if epoch >= args.epoch_num:
                break

        elif all_step_cnt >= optim_step_num:
            break

        elif last_update_time > 10:
            break

    if args.local_rank == -1 or get_rank() == 0:
        if pbar is not None:
            pbar.close()
        train_logger.close()
        eval_logger.close()


def alignment(args):
    init_args_dict = vars(args).copy()

    if args.config is not None:
        # override argparse defaults by config JSON
        opts = json.load(open(args.config))
        for k, v in opts.items():
            if isinstance(v, str):
                # PHILLY ENV special cases
                if "PHILLY_JOB_DIRECTORY" in v:
                    v = v.replace("PHILLY_JOB_DIRECTORY", os.environ["PHILLY_JOB_DIRECTORY"])
                elif "PHILLY_LOG_DIRECTORY" in v:
                    v = v.replace("PHILLY_LOG_DIRECTORY", os.environ["PHILLY_LOG_DIRECTORY"])
            setattr(args, k, v)

        # command line should override config JSON
        argv = sys.argv[1:]
        overrides, _ = parser.parse_known_args(argv)
        for k, v in vars(overrides).items():
            if f"--{k}" in argv:
                setattr(args, k, v)
        setattr(args, "local_rank", overrides.local_rank)

    if args.local_rank == -1:
        logger.info("CUDA available? {}".format(str(torch.cuda.is_available())))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = device, n_gpu
    else:
        # distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = torch.distributed.get_world_size()
        args.device, args.n_gpu = device, 1
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.local_rank == -1 or get_rank() == 0:
        logger.info("initializing cuda...")
    torch.tensor([1.], device=args.device)

    if args.local_rank == -1 or get_rank() == 0:
        logger.info("Input Argument Information")
        args_dict = vars(args)
        for a in args_dict:
            logger.info("%-28s  %s" % (a, args_dict[a]))

    preference_model_args = argparse.ArgumentParser()
    preference_model_args_dict = vars(preference_model_args)
    with open(os.path.join(args.preference_model_dir, "args.json"), "r") as f:
        summary_dict = json.load(f)
        preference_model_args_dict.update(summary_dict)
    args.mode = preference_model_args.mode

    #########################################################################
    # prepare data set
    #########################################################################
    names = {
        "inputter_name": args.inputter_name,
        "config_name": args.config_name,
    }

    toker = build_model(only_toker=True, local_rank=args.local_rank, **names)
    inputter = inputters[args.inputter_name]()

    args.dataloader_kwargs = {
        "max_input_length": args.max_input_length,
        "max_decoder_input_length": args.max_decoder_input_length,
        "max_knowledge_len": args.max_knowledge_len,
        "label_num": args.label_num,
        "only_encode": args.only_encode,
    }

    args.infer_dataloader_kwargs = {
        "max_src_turn": args.max_src_turn,
        "max_input_length": args.max_input_length,
        "max_decoder_input_length": args.max_decoder_input_length,
        "max_knowledge_len": args.max_knowledge_len,
        "label_num": args.label_num,
        "multi_knl": args.multi_knl,
        "only_encode": args.only_encode,
        "infer_batch_size": args.infer_batch_size,
    }

    _, base_model = build_model(
        checkpoint=os.path.join(args.checkpoint_dir, "pytorch_model.bin"), local_rank=args.local_rank, **names)
    base_model = deploy_model(base_model, args, local_rank=args.local_rank)

    #########################################################################
    # sampling candidates and computing preference scores
    #########################################################################
    args.candidate_dir = ".".join(args.checkpoint_dir.split(".")[:-3]) + f"_candidate_{args.candidate_num}"
    if not os.path.exists(os.path.join(args.candidate_dir, "candidates.txt")):
        logger.info(f"sampling {args.candidate_num} candidates...")
        set_seed(0)
        candidate_sampling(base_model, toker, args, **args.infer_dataloader_kwargs)
    preference_mark = args.preference_model_dir.split('/')[-1]
    if not os.path.exists(os.path.join(args.candidate_dir, f"preference_score_{preference_mark}.npy")):
        from preference_modeling import compute_scores
        logger.info(f"computing preference scores, preference model: {args.mode}...")
        compute_scores(preference_model_args, args, **args.dataloader_kwargs)

    #########################################################################
    # aligning base model with preference scores
    #########################################################################
    align_dataloader = AlignmentDataLoader(
        max_length=args.max_decoder_input_length,
        data_dir=args.candidate_dir,
        max_num=5,
        toker=toker,
        batch_size=args.train_batch_size,
        preference_mark=preference_mark,
        **names
    )

    output_dir = os.path.join("/".join(args.checkpoint_dir.split("/")[:-1]),
                              args.checkpoint_dir.split("/")[-1].split(".")[0])
    aligned_model_dir = output_dir + f"_aligned_{args.mode}_{datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')}"
    assert not os.path.exists(aligned_model_dir), f'{aligned_model_dir} has existed'
    os.mkdir(aligned_model_dir)

    if args.local_rank == -1 or get_rank() == 0:
        os.makedirs(aligned_model_dir, exist_ok=True)
        with open(os.path.join(aligned_model_dir, "alignment_args.json"), "w", encoding="utf-8") as f:
            json.dump(init_args_dict, f, ensure_ascii=False, indent=2)
        with open(os.path.join(aligned_model_dir, "custom_config.json"), "w", encoding="utf-8") as f:
            with open(f"CONFIG/{args.config_name}.json", "r", encoding="utf-8") as ff:
                json.dump(json.load(ff), f, ensure_ascii=False, indent=2)
    logger_dir = f"DATA/{args.inputter_name}.{args.config_name}"
    if not os.path.exists(logger_dir):
        os.mkdir(logger_dir)

    align_base_model(base_model, toker, align_dataloader, args, aligned_model_dir, logger)
    return aligned_model_dir


def test_checkpoints(aligned_model_dir, args):
    checkpoint_list = os.listdir(aligned_model_dir)
    checkpoint_list = [file for file in checkpoint_list if file.startswith(f"aligned_{args.config_name}_")]

    inputter = inputters[args.inputter_name]()
    args.dataloader_kwargs = {
        "max_input_length": args.max_input_length,
        "max_decoder_input_length": args.max_decoder_input_length,
        "max_knowledge_len": args.max_knowledge_len,
        "label_num": args.label_num,
        "only_encode": args.only_encode,
    }

    args.infer_dataloader_kwargs = {
        "max_src_turn": args.max_src_turn,
        "max_input_length": args.max_input_length,
        "max_decoder_input_length": args.max_decoder_input_length,
        "max_knowledge_len": args.max_knowledge_len,
        "label_num": args.label_num,
        "multi_knl": args.multi_knl,
        "only_encode": args.only_encode,
        "infer_batch_size": args.infer_batch_size,
    }

    if args.local_rank == -1:
        logger.info("CUDA available? {}".format(str(torch.cuda.is_available())))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = device, n_gpu
    else:
        # distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = torch.distributed.get_world_size()
        args.device, args.n_gpu = device, 1
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

    names = {
        "inputter_name": args.inputter_name,
        "config_name": args.config_name,
    }

    toker = build_model(only_toker=True, local_rank=args.local_rank, **names)

    for load_checkpoint_dir in checkpoint_list:
        args.checkpoint_dir = os.path.join(aligned_model_dir, load_checkpoint_dir)

        if not os.path.exists(os.path.join(args.checkpoint_dir, "inference_results")):
            set_seed(0)
            _, base_model = build_model(
                checkpoint=os.path.join(args.checkpoint_dir, "pytorch_model.bin"), local_rank=args.local_rank, **names)
            base_model = deploy_model(base_model, args, local_rank=args.local_rank)
            loss_loader = inputter.valid_dataloader(
                corpus_file=args.infer_input_file,
                toker=toker,
                batch_size=args.infer_batch_size,
                **args.infer_dataloader_kwargs
            )
            infer_dataloader = inputter.infer_dataloader(args.infer_input_file, toker, **args.infer_dataloader_kwargs)
            test_model(base_model, toker, args, loss_loader, infer_dataloader)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    #########################################################################
    # Prepare Parser
    #########################################################################

    parser = argparse.ArgumentParser()
    # model configuration
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--inputter_name", type=str, required=True)
    parser.add_argument("--align_seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", "-c", type=str, default=None)
    parser.add_argument("--preference_model_dir", type=str,
                        default="../preference_modeling/output/esc_d-PM_23-0527-2227_fold1")
    parser.add_argument("--save_total_limit", type=int, default=3)

    # base model data processing --> for testing and candidate sampling
    parser.add_argument("--max_src_turn", type=int, default=None)
    parser.add_argument("--max_input_length", type=int, default=160)
    parser.add_argument("--max_decoder_input_length", type=int, default=40)
    parser.add_argument("--max_knowledge_len", type=int, default=None)
    parser.add_argument("--multi_knl", action="store_true", help="allow candidate knowledge items")
    parser.add_argument("--label_num", type=int, default=None)
    parser.add_argument("--only_encode", action="store_true", help="only do encoding")

    # infer dataset and candidate data path
    parser.add_argument("--train_input_file", type=str, default="DATA/train.txt")
    parser.add_argument("--eval_input_file", type=str, default="DATA/valid.txt")
    parser.add_argument("--infer_input_file", type=str, default="DATA/test.txt")
    parser.add_argument("--candidate_num", type=int, default=10)

    # training hyper parameters for alignment
    parser.add_argument("--epoch_num", type=int, default=10, help="how many training epochs")
    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size now means per GPU per step")
    parser.add_argument("--accumulate_step", type=int, default=16,
                        help="to increase effective batch size and reduce synchronization")
    parser.add_argument("--valid_step", type=int, default=400)
    parser.add_argument("--max_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_step", type=int, default=100, help="new API specifies num align steps")

    parser.add_argument("--margin", type=float, default=0.001, help="margin for ranking loss on candidate responses")
    parser.add_argument("--gold_margin", type=float, default=0, help="margin for ranking loss on gold responses")
    parser.add_argument("--gold_weight", type=float, default=0, help="weight for ranking loss on gold responses")
    parser.add_argument("--rank_weight", type=float, default=1, help="weight for ranking loss on candidate responses")
    parser.add_argument("--lm_weight", type=float, default=1, help="weight for mle loss on gold responses")
    parser.add_argument("--scale", type=float, default=1, help="scale of ranking loss")

    parser.add_argument("--infer_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    parser.add_argument("--optim_step_num", type=int, default=20000, help="new API specifies num align steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", type=boolean_string, default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument("--pbar", type=boolean_string, default=True, help="turn on progress bar")
    parser.add_argument("--chinese", action="store_true", help="chinese language")

    # distributed
    parser.add_argument("--local_rank", type=int, default=-1, help="for torch.distributed")
    parser.add_argument("--config", help="JSON config file")

    # do normal parsing
    args = parser.parse_args()

    assert args.checkpoint_dir is not None, "``checkpoint_dir`` is invalid!"
    assert args.preference_model_dir is not None, "``preference_model_dir`` is invalid!"
    aligned_model_dir = alignment(args)
    test_checkpoints(aligned_model_dir, args)
