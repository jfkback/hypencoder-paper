from typing import List, Optional

import fire
import torch
from transformers import AutoTokenizer

from hypencoder_cb.inference.shared import (
    BaseEncoder,
    encode_ir_dataset_items_to_disk,
    encode_jsonl_items_to_disk,
    VectorEncodedRepresentation
)
from hypencoder_cb.modeling.hypencoder import (
    HypencoderDualEncoder,
    TextDualEncoder,
)
from hypencoder_cb.utils.torch_utils import dtype_lookup


class InferenceTextEncoder(BaseEncoder):
    encoding_type = VectorEncodedRepresentation

    def __init__(
        self,
        model_name_or_path: str,
        model_type: str = "hypencoder_dual_encoder",
        device: str = "cuda",
        max_length: int = 512,
        dtype: str = "fp32",
    ) -> None:
        """
        Args:
            model_name_or_path (str): The name or path of the model to use.
                This should point to either a HypencoderDualEncoder or a
                TextDualEncoder checkpoint.
            model_type (str, optional): The type of model to use. The options
                are "hypencoder_dual_encoder" or "text_dual_encoder". Defaults
                to "hypencoder_dual_encoder".
            device (str, optional): The device to use. Defaults to "cuda".
            max_length (int, optional): Max length of the text being encoded.
                Defaults to 512.
            dtype (str, optional): The dtype to use. Options are "fp16",
                "fp32", and "bf16". Defaults to "fp32".
        """

        self.dtype = dtype_lookup(dtype)

        model_cls = {
            "hypencoder_dual_encoder": HypencoderDualEncoder,
            "text_dual_encoder": TextDualEncoder,
        }[model_type]

        self.model = (
            model_cls.from_pretrained(model_name_or_path)
            .passage_encoder.to(device, dtype=self.dtype)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = device
        self.max_length = max_length

    def batch_encode(self, texts: List[str]) -> VectorEncodedRepresentation:
        """

        Args:
            texts (List[str]): A list of texts to encode.

        Returns:
            VectorEncodedRepresentation: The encoded representations of the
                input texts. The shape of the output is (batch_size, dim).
        """

        tokenized_texts = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=tokenized_texts["input_ids"],
                attention_mask=tokenized_texts["attention_mask"],
            )

            return output


def do_encoding(
    model_name_or_path: str,
    output_path: str,
    jsonl_path: Optional[str] = None,
    ir_dataset_name: Optional[str] = None,
    item_id_key: str = "id",
    item_text_key: str = "text",
    batch_size: int = 512,
    model_type: str = "hypencoder_dual_encoder",
    max_length: int = 512,
    dtype: str = "fp32",
) -> None:
    """Encodes a dataset of items to disk using the specified model.

    Args:
        model_name_or_path (str): The name or path of the model to use.
                This should point to either a HypencoderDualEncoder or a
                TextDualEncoder checkpoint.
        output_path (str): The output path to save the encoded items to.
        jsonl_path (Optional[str], optional): If provided this is used as the
            input to the encoder, must have the keys `item_id_key` and
            `item_text_key`. If None, `ir_dataset_name` should be provided.
            Defaults to None.
        ir_dataset_name (Optional[str], optional): If provided the documents
            are used as the input to the encoder. If None `jsonl_path` should
            be provided. Defaults to None.
        item_id_key (str, optional): When using `jsonl_path` this is the key
            of the item ID in the JSONL. Defaults to "id".
        item_text_key (str, optional): When using `jsonl_path` this is the key
            of the item text in the JSONL. Defaults to "text".
        batch_size (int, optional): The batch size to use while encoding.
            Defaults to 512.
        model_type (str, optional): See `InferenceTextEncoder` for details.
            Defaults to "hypencoder_dual_encoder".
        max_length (int, optional): Max length of text input to encoder.
            Defaults to 512.
        dtype (str, optional):  See `InferenceTextEncoder` for details.
            Defaults to "fp32".

    Raises:
        ValueError: If both `jsonl_path` and `ir_dataset_name` are provided.
    """

    if jsonl_path is not None and ir_dataset_name is not None:
        raise ValueError(
            "Only one of jsonl_path and ir_dataset_name can be provided."
        )

    encoder = InferenceTextEncoder(
        model_name_or_path=model_name_or_path,
        model_type=model_type,
        max_length=max_length,
        dtype=dtype,
    )

    if jsonl_path is not None:
        encode_jsonl_items_to_disk(
            encoder=encoder,
            input_path=jsonl_path,
            output_path=output_path,
            batch_size=batch_size,
            item_id_key=item_id_key,
            item_text_key=item_text_key,
        )
    else:
        encode_ir_dataset_items_to_disk(
            encoder=encoder,
            ir_dataset_name=ir_dataset_name,
            output_path=output_path,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    fire.Fire(do_encoding)
