## Alignment of MultiESC

Our implementation is based on [MultiESC](https://github.com/lwgkzl/MultiESC/tree/main/MultiESC/data).

### Data Download

Download ``NRC_VAD.txt``, ``train.txt``, ``valid.txt`` and ``test.txt``
from [MultiESC](https://github.com/lwgkzl/MultiESC/tree/main/MultiESC/data).
Put them in the folder `data`.

### Data Preprocessing \& Base Model Training

Following the instructions in [MultiESC](https://github.com/lwgkzl/MultiESC/tree/main).

### Aligned Model Training

Run:

```console
CUDA_VISIBLE_DEVICES=0,1 nohup python align_MultiESC.py --data_type=8 --model_name_or_path=./final_output/lwg_whlookahead_generate --learning_rate=5e-5 --lr2=1e-4 --with_cause --with_strategy --lookahead --model_type=1 --candidate_num=10 --preference_model_dir ../preference_modeling/output/esc_d-PM_23-0527-2227_fold1 --per_device_train_batch_size=6 > align_MultiESC 2>&1 &
```

Change ``--preference_model_dir`` and ``--model_name_or_path`` to the path of the preference model checkpoint folder and
the base model checkpoint folder, respectively.
