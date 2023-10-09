## Alignment of ESConv (Blender-Vanilla & Blender-Joint)

Our implementation is based on [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation/tree/main/codes_zcj).

### Data Download

Download [ESConv.json](https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json)
and [strategy.json](https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/strategy.json) and
put them in the folder `DATA`.

### Dara Preprocessing

Enter `DATA` and run ``python process.py``.

To preprocess the training data (for Blender-Vanilla and Blender-Joint, respectively), run:

```console
python prepare.py --config_name vanilla --inputter_name vanilla --train_input_file DATA/train.txt --max_input_length 160 --max_decoder_input_length 40
```

```console
python prepare.py  --config_name strat --inputter_name strat --train_input_file DATA/train.txt --max_input_length 160  --max_decoder_input_length 40
```

### Base Model Training

Run:

```console
. RUN/train_vanilla.sh {gpu_id}
```

```console
. RUN/train_strat.sh {gpu_id}
```

### Aligned Model Training

Change the value of ``--preference_model_dir`` in the **RUN/align_vanilla.sh** and **RUN/align_strat** files.
The``--preference_model_dir`` is the path of the preference model checkpoint folder.
Run:

```console
. RUN/align_vanilla.sh {gpu_id} {model_path}
```

```console
. RUN/align_strat.sh {gpu_id} {model_path}
```

The ``{model_path}`` is the path of the base model checkpoint folder.

### Model Inference

Run:

```console
. RUN/infer_model.sh {gpu_id} {model_name} {model_path}
```

``{model_name}`` can be either **vanilla** or **strat**.




