import os
import json
import time
import argparse
import datetime
import sys

sys.path.append("../")
from os.path import join

import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.distributed import get_rank, get_world_size
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed

from preference_modeling.inputters import inputters
from preference_modeling.utils.building_utils import build_model, deploy_model
from preference_modeling.utils.distributed import all_reduce_and_rescale_tensors, all_gather_list
from preference_modeling.utils.eval_utils import eval_model
import warnings

warnings.filterwarnings("ignore")

INF = 100000000
CACHE_EMPTY_STEP = 10000
SAVE_DIR = "output"
if os.path.exists(SAVE_DIR) == False:
    os.mkdir(SAVE_DIR)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)

    #########################################################################
    # Prepare Parser
    ##########################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="esc", choices=["esc", "mic", "off"])
    parser.add_argument("--mode", type=str, default="d-PM", choices=["d-PM", "soft", "major", "single"])
    parser.add_argument("--fold_num", type=int, default=1)
    parser.add_argument("--class_num", type=int, default=3)
    parser.add_argument("--off_fine_task", type=str, default="offensive", choices=["offensive", "hate", "aggressive"])

    parser.add_argument("--preseqlen", type=int, default=10)
    parser.add_argument("--d_prefix", type=int, default=512)
    parser.add_argument("--prefix_dropout", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epoch_num", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--valid_step", type=int, default=800)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=400)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--loss_scale", type=float, default=0.0)
    parser.add_argument("--pbar", type=bool, default=True)
    parser.add_argument("--chinese", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    args = parser.parse_args()
    parser = argparse.ArgumentParser()
    try:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    except KeyError:
        args.local_rank = -1

    if args.task_name == 'esc':
        assert args.fold_num >= 0, 'fold_num must be >= 0!'
    else:
        args.fold_num = -1
        args.class_num = 2

    init_args_dict = vars(args).copy()

    if args.local_rank == -1:
        logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = device, n_gpu
    else:
        # distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = device, 1
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}"
                    .format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    logger.info("Done!")
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, 'batch size % gradient accumulation steps != 0!'
    args.train_batch_size = (args.train_batch_size // args.gradient_accumulation_steps)
    if args.local_rank == -1 or get_rank() == 0:
        logger.info('train batch size = {}, new train batch size (after gradient accumulation) = {}'.format(
            args.train_batch_size * args.gradient_accumulation_steps, args.train_batch_size))

    if args.local_rank == -1 or get_rank() == 0:
        logger.info('initializing cuda...')
    torch.tensor([1.], device=args.device)
    if args.local_rank == -1 or get_rank() == 0:
        logger.info('Input Argument Information')
        args_dict = vars(args)
        for a in args_dict:
            logger.info('%-28s  %s' % (a, args_dict[a]))

    set_seed(args.seed)
    #########################################################################
    # Prepare Data Set
    ##########################################################################
    if args.mode == "single":
        task_name = "single"
    else:
        task_name = args.task_name
    names = {
        "mode": args.mode,
        "inputter_name": task_name,
        "fold_num": args.fold_num,
        "fine_task": args.off_fine_task,
    }
    toker = build_model(only_toker=True)
    inputter = inputters[task_name]()

    if args.local_rank == -1:
        train_dataloader = inputter.train_dataloader(
            toker=toker,
            feature_dataset=inputter.train_dataset,
            batch_size=args.train_batch_size,
            **names
        )

    else:
        train_dataloader = inputter.train_distributed_dataloader(
            get_rank(),
            get_world_size(),
            toker=toker,
            feature_dataset=inputter.train_dataset,
            batch_size=args.train_batch_size,
            **names
        )

    if args.epoch_num != None:
        args.num_optim_steps = args.epoch_num * (len(train_dataloader) // args.train_batch_size
                                                 + int(len(train_dataloader) % args.train_batch_size != 0))

    eval_dataloader_loss = inputter.valid_dataloader(
        toker=toker,
        batch_size=args.eval_batch_size,
        **names
    )

    #########################################################################
    # Prepare Model and Optimizer
    #########################################################################
    load_checkpoint = args.load_checkpoint
    _, model = build_model(checkpoint=load_checkpoint, args=args, **names)
    model = deploy_model(model, args)

    if args.local_rank != -1:
        # when from scratch make sure initial models are the same
        params = [p.data for p in model.parameters()]
        all_reduce_and_rescale_tensors(params, float(torch.distributed.get_world_size()))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    if args.local_rank == -1 or get_rank() == 0:
        logger.info('Number of parameter = {}'.format(total_params))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']  # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_optim_steps
    )

    if args.fp16:
        logger.info('in fp16, using FusedAdam')
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    #########################################################################
    # Training !
    #########################################################################

    timestamp = datetime.datetime.now().strftime('%Y-%m%d-%H%M')[2:]
    output_dir_str = f"{args.task_name}_{args.mode}_{timestamp}"
    if args.fold_num > 0:
        output_dir_str += f"_fold{args.fold_num}"
    if args.task_name == "off":
        output_dir_str += f"_{args.off_fine_task}"
    output_dir = join(SAVE_DIR, output_dir_str)
    if args.local_rank == -1 or get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        logger.info('output_dir = {}'.format(output_dir))
        with open(join(output_dir, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(init_args_dict, f, ensure_ascii=False, indent=2)
        with open(join(output_dir, 'custom_config.json'), 'w', encoding='utf-8') as f:
            with open(f'preference_model.json', 'r', encoding='utf-8') as ff:
                json.dump(json.load(ff), f, ensure_ascii=False, indent=2)

    if args.local_rank == -1 or get_rank() == 0:
        train_logger = open(join(output_dir, 'train_log.csv'), 'a+', buffering=1)
        eval_logger = open(join(output_dir, 'eval_log.csv'), 'a+', buffering=1)
        print('epoch\tglobal_step\tstep\ttmp_loss\tmean_loss\tepoch_time', file=train_logger)
        print('epoch\tglobal_step\tstep\tfreq_loss', file=eval_logger)

    global_step = 0
    lowest_eval_loss = 1
    step = 0
    epoch = 0

    if args.local_rank == -1 or get_rank() == 0:
        pbar = tqdm(total=args.num_optim_steps, desc=f"training", position=0)

    while True:
        model.train()
        tr_loss, mean_loss, nb_tr_examples, nb_tr_steps = 0.0, 0.0, 0, 0
        train_start_time_epoch = time.time()
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

            batch.update({'global_step': global_step})
            batch.update({'epoch': epoch})
            batch.update({'warmup_steps': args.warmup_steps})
            outputs = model(**batch)

            loss = outputs["loss"]
            predictions = outputs["dist"].cpu()
            if 'input_ids' in batch:
                input_ids = batch['input_ids']
            elif 'tgt_input_ids' in batch:
                input_ids = batch['tgt_input_ids']
            else:
                assert 'src_input_ids' in batch
                input_ids = batch['src_input_ids']

            if n_gpu > 1:
                loss = loss.mean()
            loss = loss / (args.train_batch_size * args.gradient_accumulation_steps / input_ids.shape[0])

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tmp_loss = float(loss.item()) * (
                    args.train_batch_size * args.gradient_accumulation_steps / input_ids.shape[0])
            tr_loss += tmp_loss
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            mean_loss = tr_loss / nb_tr_steps

            # gradient update
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.local_rank != -1:
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad != None]
                    all_reduce_and_rescale_tensors(grads, float(1))

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # Print log info to file
                if args.local_rank != -1:
                    mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()

                if args.local_rank == -1 or get_rank() == 0:
                    epoch_time = time.time() - train_start_time_epoch
                    pbar_str = ''  # f"tok/s: {n_token_real_all_proc//epoch_time//1000}k "
                    pbar_str += f"loss: {tmp_loss:.2f} "
                    pbar_str += f"mean loss: {mean_loss:.2f} epoch: {epoch + 1}"

                    pbar.set_postfix_str(pbar_str)
                    if args.epoch_num != None:
                        pbar.update(args.gradient_accumulation_steps)
                    else:
                        pbar.update(1)

                print(f'{epoch + 1}\t{global_step}\t{step}\t{tmp_loss}\t{mean_loss}\t{epoch_time}',
                      file=train_logger)

                if global_step % args.valid_step == 0:  # and epoch > 0:
                    if args.local_rank == -1 or get_rank() == 0:
                        # only rank 0 process evaluate
                        eval_loss, *_, = eval_model(
                            model=model,
                            eval_dataloader=eval_dataloader_loss,
                            infer=False,
                            args=args,
                        )
                        logger.info(f'**Eval (step {global_step})** eval_loss: {eval_loss}')
                        print(f'{epoch + 1}\t{global_step}\t{step + 1}\t{eval_loss}', file=eval_logger)
                        logger.info('current learning rate: ' + str(optimizer.param_groups[0]['lr']))
                        if eval_loss < lowest_eval_loss:
                            torch.save(model.state_dict(), join(output_dir, f'model_step-{global_step}.bin'))
                            torch.save(model.state_dict(), join(output_dir, 'best.bin'))
                            toker.save_vocabulary(output_dir)
                            try:
                                model.config.to_json_file(join(output_dir, f'config.json'))
                            except AttributeError:
                                model.module.config.to_json_file(join(output_dir, f'config.json'))
                            lowest_eval_loss = eval_loss
                        model.train()

                if args.epoch_num is None and global_step >= args.num_optim_steps:
                    break

            if (step + 1) % CACHE_EMPTY_STEP == 0:
                torch.cuda.empty_cache()
        epoch += 1
        # if args.epoch_num != None:
        #     if args.local_rank == -1 or get_rank() == 0:
        #         # only rank 0 process evaluate
        #         eval_loss, *_, = eval_model(
        #             model=model,
        #             eval_dataloader=eval_dataloader_loss,
        #             infer=False,
        #             args=args,
        #         )
        #         logger.info(f'**Eval (epoch {epoch})** eval_loss: {eval_loss}')
        #         print(f'{epoch}\t{global_step}\t{step}\t{eval_loss}', file=eval_logger)
        #         logger.info('current learning rate: ' + str(optimizer.param_groups[0]['lr']))
        #         if eval_loss < lowest_eval_loss:
        #             torch.save(model.state_dict(), join(output_dir, f'model_epoch-{epoch}.bin'))
        #             torch.save(model.state_dict(), join(output_dir, 'best.bin'))
        #             toker.save_vocabulary(output_dir)
        #             try:
        #                 model.config.to_json_file(join(output_dir, 'config.json'))
        #             except AttributeError:
        #                 model.module.config.to_json_file(join(output_dir, 'config.json'))
        #             lowest_eval_loss = eval_loss
        #         model.train()

        if args.epoch_num is None and global_step >= args.num_optim_steps:
            break

        if args.epoch_num != None and epoch == args.epoch_num:
            break

    if args.local_rank == -1 or get_rank() == 0:
        if pbar is not None:
            pbar.close()
        train_logger.close()
        eval_logger.close()
