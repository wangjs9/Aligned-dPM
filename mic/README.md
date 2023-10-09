## Alignment of RoT Generation Models

Our implementation is based on [MIC.csv](https://github.com/SALT-NLP/mic).

### Training RoT Generation Model

Download dataset([MIC.csv](https://github.com/SALT-NLP/mic)) and put it in the **data/mic** folder.

```console
. RUN/rot_run.sh
```

By default, the model will be saved in the **rot_generation/output** folder.

### Aligning RoT Generation Model

Change the value of ``--input`` and ``--annotator_load_checkpoint`` in the **RUN/rot_align_run.sh** file.
The ``--input`` is the path of the RoT generation model checkpoint folder, and the ``--annotator_load_checkpoint`` is
the path of the annotator checkpoint folder.

```console
. RUN/rot_align_run.sh
```

By default, the model will be saved in the **align_rot_generation/output** folder.

### RoT Generation (Inference on Test Set)

```console
. RUN/decode_all.sh "{foler_name}/output/{folder_prefix}*" "./data/mic/MIC.csv" "Q [answ] A [rot] ~ rot" {gpu_id} {seed}
```

``{folder_name}`` can be either **align_rot_generation** or **rot_generation**.

### Evaluation Metric Computation

```console
python -m {foler_name}.metrics --input "align_rot_generation/output/*" --output "all_results.csv"
```

{folder_name} can be either **align_rot_generation** or **rot_generation**.