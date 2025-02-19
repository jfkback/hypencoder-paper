import os
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import fire
from omegaconf import OmegaConf


@dataclass
class HypencoderModelConfig:
    tokenizer_pretrained_model_name_or_path: Optional[str] = None

    query_encoder_kwargs: Dict = field(default_factory=dict)
    passage_encoder_kwargs: Dict = field(default_factory=dict)

    # Union[str, List[str]]
    loss_type: Any = field(default_factory=lambda: [])
    # Union[Dict[str, Any], List[Dict[str, Any]]]
    loss_kwargs: Any = field(default_factory=list)

    checkpoint_path: Optional[str] = None

    model_type: Literal["hypencoder", "biencoder"] = "hypencoder"
    shared_encoder: bool = False


@dataclass
class HypencoderDataConfig:
    training_data_jsonl: Optional[str] = None
    validation_data_jsonl: Optional[str] = None

    training_huggingface_dataset: Optional[str] = None
    validation_huggingface_dataset: Optional[str] = None

    training_data_split: str = "train"
    validation_data_split: str = "train"

    positive_filter_type: str = "first"
    positive_filter_kwargs: Optional[Dict[str, Any]] = None

    label_key: Optional[str] = "score"

    num_positives_to_sample: int = 1
    num_negatives_to_sample: int = 7


@dataclass
class HFTrainerConfig:
    output_dir: str = ""
    overwrite_output_dir: bool = False
    remove_unused_columns: bool = False

    evaluation_strategy: str = "no"
    eval_strategy: str = "no"
    eval_steps: int = 500

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 0
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: Optional[int] = None
    ignore_data_skip: bool = False

    learning_rate: float = 5e-5
    weight_decay: float = 0.0

    num_train_epochs: Optional[int] = 1
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.05
    warmup_steps: int = 0

    logging_strategy: str = "steps"
    logging_steps: int = 1
    max_steps: int = -1

    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    save_only_model: bool = False
    save_safetensors: bool = True

    bf16: bool = False
    fp16: bool = False
    tf32: bool = False
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    run_name: Optional[str] = None
    disable_tqdm: bool = False

    ddp_find_unused_parameters: Optional[bool] = True
    # str or bool string options are: "full_shard", "auto_wrap", ...
    fsdp: Any = False
    fsdp_config: Optional[Dict[str, Any]] = None

    report_to: str = "none"

    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_private_repo: bool = True
    gradient_checkpointing: bool = False

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


@dataclass
class HypencoderTrainerConfig:
    hf_trainer_config: HFTrainerConfig = HFTrainerConfig(
        output_dir="/tmp/output/"
    )
    resume_from_checkpoint: Optional[Any] = False


@dataclass
class HypencoderTrainingConfig:
    model_config = HypencoderModelConfig(
    )
    data_config = HypencoderDataConfig(
        training_data_jsonl="",
    )
    trainer_config = HypencoderTrainerConfig(
        hf_trainer_config=HFTrainerConfig(),
    )


def relative_file_path_to_abs_path(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)


def export_config_to_yaml(
    config_name: Optional[str] = None,
    config_dir: str = "configs",
) -> None:
    config_dir = relative_file_path_to_abs_path(config_dir)
    config = OmegaConf.structured(HypencoderTrainingConfig)

    if config_name is None:
        config_name = config.trainer_config.hf_trainer_config.run_name

    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    config_path = os.path.join(config_dir, config_name)

    if os.path.isfile(config_path):
        raise ValueError(
            f"The config file {config_path} already exists. Please choose a"
            " different config name."
        )

    print(f"Exporting config to {config_path}")
    OmegaConf.save(config=config, f=config_path)


if __name__ == "__main__":
    fire.Fire(export_config_to_yaml)
