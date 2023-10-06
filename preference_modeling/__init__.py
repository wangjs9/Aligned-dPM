import os
import numpy as np
from torch import Tensor
from preference_modeling.inputters import inputters
from preference_modeling.utils.building_utils import build_model, deploy_model


def compute_scores(preference_model_args, args, **dataloader_kwargs):
    mode = preference_model_args.mode
    kwargs = {'mode': mode}
    toker, model = build_model(
        checkpoint=os.path.join(args.preference_model_dir, "best.bin"),
        local_rank=args.local_rank,
        args=preference_model_args,
        **kwargs,
    )
    deploy_model(model, args)
    model.eval()

    input_file = os.path.join(args.candidate_dir, "candidates.txt")
    preference_mark = args.preference_model_dir.split('/')[-1]
    save_path = os.path.join(args.candidate_dir, f"preference_score_{preference_mark}.npy")

    inputter = inputters["esc"]()
    with open(input_file, 'r', encoding='UTF-8') as f:
        corpus = f.readlines()
    infer_dataloader = inputter.infer_dataloader(
        toker=toker,
        corpus=corpus,
        batch_size=args.eval_batch_size,
        **dataloader_kwargs,
    )

    predictions = []
    for batch, sample_idx in infer_dataloader:
        batch = {k: v.to(args.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        scores = model.predict(**batch)[:, -1].cpu().numpy()
        predictions.append(scores)

    np.save(save_path, predictions)


class preference_pipe(object):
    def __init__(self, preference_model_args, args, **dataloader_kwargs):
        mode = preference_model_args.mode
        kwargs = {'mode': mode}
        self.toker, self.model = build_model(
            checkpoint=os.path.join(args.preference_model_dir, "best.bin"),
            args=preference_model_args,
            **kwargs,
        )
        deploy_model(self.model, args)
        self.model.eval()
        self.inputter = inputters["esc"]()
        self.dataloader_kwargs = dataloader_kwargs
        self.dataloader_kwargs.update({'no_bar_info': True})

    def compute_score(self, text_batch):
        infer_dataloader = self.inputter.infer_dataloader(
            toker=self.toker,
            corpus=text_batch,
            batch_size=1,
            **self.dataloader_kwargs,
        )

        predictions = []
        for batch, sample_idx in infer_dataloader:
            batch = {k: v.to(self.model.device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            scores = self.model.predict(**batch)[:, -1].cpu()
            predictions.append(scores)
        return predictions
