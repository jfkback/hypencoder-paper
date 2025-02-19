
## Overview
The main training code is in `train.py`. It uses arguments from `args.py` which are passed as a yaml config file. To customize training you can edit the yaml configuration file.

To run training call the training script with a path to a training configuration like so:
```
python hypencoder_cb/train/train.py hypencoder_cb/train/configs/hypencoder.6_layer.yaml
```
for multi-gpu training it is recommended to use either torch-launch or Huggingface's accelerate launch like this:
```
accelerate launch --num_processes={{num_gpus}} --main_process_port={{main_process_port}} ypencoder_cb/train/train.py hypencoder_cb/train/configs/hypencoder.6_layer.yaml
```


## Reproducing Paper Results
The configurations for the models in the paper are in the `configs` directory and have the name `hypencoder.[NUM_LAYERS]_layers.yaml`. Where valid `NUM_LAYERS=[2, 4, 6, 8]`. To run them follow the steps in the overview.

Note, that the training runs in the paper had 