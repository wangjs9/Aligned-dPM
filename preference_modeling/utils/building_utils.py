# coding=utf-8

import json
import os
import logging
import torch

try:
    from preference_modeling.models import models
except ModuleNotFoundError:
    from models import models
from transformers import (AutoTokenizer, AutoModel, AutoConfig)

logger = logging.getLogger(__name__)


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def build_model(only_toker=False, checkpoint=None, args=None, **kwargs):
    config_file = "/home/jiashuo/codes/Aligned-dPM/preference_modeling/preference_model.json"
    if not os.path.exists(config_file):
        raise ValueError

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if 'model_name' not in config or 'pretrained_model_path' not in config:
        raise ValueError
    toker = AutoTokenizer.from_pretrained(config['pretrained_model_path'])

    if only_toker:
        if 'expanded_vocab' in config:
            toker.add_tokens(config['expanded_vocab'], special_tokens=True)
        return toker

    assert "mode" in kwargs
    mode = kwargs.pop("mode")
    Model = models[mode]
    model = Model.from_pretrained(config['pretrained_model_path'], args)
    model.classifier = None
    for name, param in model.named_parameters():
        if name.startswith('bert.'):
            param.requires_grad = False

    if config.get('custom_config_path', None) != None:
        model = model(AutoConfig.from_pretrained(config['custom_config_path']))

    if 'gradient_checkpointing' in config:
        setattr(model.config, 'gradient_checkpointing', config['gradient_checkpointing'])

    if 'expanded_vocab' in config:
        toker.add_tokens(config['expanded_vocab'], special_tokens=True)
    model.tie_tokenizer(toker)

    if checkpoint != None:
        logger.info('loading finetuned model from %s' % checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')), strict=False)

    return toker, model


def load_model(model, checkpoint, local_rank=-1):
    if checkpoint != None and checkpoint.lower() != "none":
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict, local_rank)
        logger.info('loading finetuned model from %s' % checkpoint)

        strict = False

        needed_keys = set(dict(model.named_parameters()).keys())
        loaded_keys = []
        for k, v in model_state_dict.items():
            if k not in needed_keys:
                continue
            try:
                model.load_state_dict({k: v}, strict=False)
                loaded_keys.append(k)
            except RuntimeError as e:
                logger.info(' ??? unmatched parameter [%s]' % k)
                if strict:
                    raise e

        loaded_keys = set(loaded_keys)
        missed_keys = needed_keys - loaded_keys

        if len(missed_keys) > 0:
            for k in sorted(missed_keys):
                logger.info(' !!! parameter [%s] missed' % k)


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in list(model_state_dict.keys()).copy():
        new_key = t

        if new_key.startswith('module.'):
            new_key = new_key.replace('module.', '')
        elif new_key.startswith('model.'):
            new_key = new_key.replace('model.', '')

        if new_key.endswith('.beta'):
            new_key = new_key.replace('.beta', '.bias')
        elif new_key.endswith('.gamma'):
            new_key = new_key.replace('.gamma', '.weight')

        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict


def deploy_model(model, args):
    logger.info('deploying preference model...')
    n_gpu = args.n_gpu
    device = args.device

    if n_gpu > 1:
        logging.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)

    model.to(device)

    return model
