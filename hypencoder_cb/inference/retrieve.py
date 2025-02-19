from pathlib import Path
from typing import Dict, List, Optional, Union

import fire
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from hypencoder_cb.inference.shared import (
    BaseRetriever,
    Item,
    TextQuery,
    load_encoded_items_from_disk,
    retrieve_for_ir_dataset_queries,
    retrieve_for_jsonl_queries,
)
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder
from hypencoder_cb.utils.data_utils import (
    load_qrels_from_ir_datasets,
    load_qrels_from_json,
)
from hypencoder_cb.utils.eval_utils import (
    calculate_metrics_to_file,
    load_standard_format_as_run,
    pretty_print_standard_format,
)
from hypencoder_cb.utils.iterator_utils import batchify_slicing
from hypencoder_cb.utils.torch_utils import dtype_lookup


class HypencoderRetriever(BaseRetriever):

    def __init__(
        self,
        model_name_or_path: str,
        encoded_item_path: str,
        batch_size: int = 100_000,
        device: str = "cuda",
        dtype: Union[torch.dtype, str] = "float32",
        query_model_kwargs: Optional[Dict] = None,
        put_all_embeddings_on_device: bool = True,
        query_max_length: int = 32,
        ignore_same_id: bool = False,
    ) -> None:
        """
        Args:
            model_name_or_path (str): Name or path to a HypencoderDualEncoder
                checkpoint.
            encoded_item_path (str): Path to the encoded items.
            batch_size (int, optional): Batch sized used for scoring. Defaults
                to 100,000.
            device (str, optional): The device to use. Defaults to "cuda".
            dtype (Union[torch.dtype, str], optional): The dtype to use for the
                model and embedded items. Options are "fp16", "fp32", and
                "bf16". Defaults to "float32".
            query_model_kwargs (Optional[Dict], optional): Key-word arguments
                passed to the q-net in addition to the item representations.
                Defaults to None.
            put_all_embeddings_on_device (bool, optional): Whether all
                embeddings should be put on the device. If False, all
                embeddings are kept in RAM instead of in VRAM. It is faster
                with this set to True, but it requires more GPU memory.
                Defaults to True.
            query_max_length (int, optional): Maximum length of the query.
                Defaults to 32.
            ignore_same_id (bool, optional): Whether to ignore retrievals
                with the same ID as the query. This is only relevant for
                certain datasets. Defaults to False.
        """
        if isinstance(dtype, str):
            dtype = dtype_lookup(dtype)

        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.encoded_item_path = encoded_item_path
        self.query_max_length = query_max_length
        self.ignore_same_id = ignore_same_id
        self.put_on_device = put_all_embeddings_on_device

        if query_model_kwargs is None:
            query_model_kwargs = {}

        self.query_model_kwargs = query_model_kwargs

        self.model = (
            HypencoderDualEncoder.from_pretrained(model_name_or_path)
            .to(device, dtype=self.dtype)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        print("Started loading encoded items...")
        encoded_items = load_encoded_items_from_disk(
            encoded_item_path,
        )

        self.encoded_item_embeddings = torch.stack(
            [
                torch.tensor(x.representation, dtype=self.dtype)
                for x in tqdm(encoded_items)
            ]
        )

        if self.put_on_device:
            self.encoded_item_embeddings = self.encoded_item_embeddings.to(
                self.device
            )

        self.encoded_item_ids = [x.id for x in tqdm(encoded_items)]
        self.encoded_item_texts = [x.text for x in tqdm(encoded_items)]

    def retrieve(self, query: TextQuery, top_k: int) -> List[Item]:
        tokenized_query = self.tokenizer(
            query.text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.query_max_length,
        ).to(self.device)

        with torch.no_grad():
            query_output = self.model.query_encoder(
                input_ids=tokenized_query["input_ids"],
                attention_mask=tokenized_query["attention_mask"],
            )

        query_model = query_output.representation

        num_batches = (
            len(self.encoded_item_embeddings) // self.batch_size
        ) + 1

        top_k_indices = torch.full((top_k * num_batches,), -1)
        top_k_scores = torch.full((top_k * num_batches,), -float("inf"))

        for batch_index, batch_item_embeddings in enumerate(
            batchify_slicing(self.encoded_item_embeddings, self.batch_size)
        ):
            if not self.put_on_device:
                batch_item_embeddings = batch_item_embeddings.to(self.device)

            batch_item_embeddings = batch_item_embeddings.unsqueeze(0)

            similarity_matrix = query_model(
                batch_item_embeddings, **self.query_model_kwargs
            ).squeeze()

            values, indices = torch.topk(similarity_matrix, top_k, dim=0)
            indices = indices.squeeze(0).cpu()
            values = values.squeeze(0).cpu()

            top_k_indices[batch_index * top_k : (batch_index + 1) * top_k] = (
                indices + (batch_index * self.batch_size)
            )
            top_k_scores[batch_index * top_k : (batch_index + 1) * top_k] = (
                values
            )

        final_values, indices = torch.topk(top_k_scores, top_k, dim=0)
        final_indices = top_k_indices[indices]

        items = []
        for item_idx, score in zip(final_indices, final_values):
            if (
                self.ignore_same_id
                and query.id == self.encoded_item_ids[item_idx]
            ):
                continue

            items.append(
                Item(
                    text=self.encoded_item_texts[item_idx],
                    id=self.encoded_item_ids[item_idx],
                    score=score.item(),
                    type="hypencoder_retriever",
                )
            )

        return items


def do_eval_and_pretty_print(
    retrieval_path: str,
    output_dir: str,
    ir_dataset_name: Optional[str] = None,
    qrel_json: Optional[str] = None,
    metric_names: Optional[List[str]] = None,
) -> None:
    """Does evaluation and pretty prints the retrieval results for easier
    inspection.

    Args:
        retrieval_path (str): Path to the retrieval JSONL file.
        output_dir (str): Path to the output directory.
        ir_dataset_name (Optional[str], optional): If provided is used to
            get the qrels used for evaluation. If None, then `qrel_json` must
            be provided. Defaults to None.
        qrel_json (Optional[str], optional): If provided is used as the qrels
            for evaluation. If None, then `qrel_json` must
            be provided. Defaults to None.
        metric_names (Optional[List[str]], optional): A list of metrics to
            compute. These are passed to IR-Measures so should be compatible.
            If None, a default set of metrics is found. Defaults to None.

    Raises:
        ValueError: If both `ir_dataset_name` and `qrel_json` are provided.
        ValueError: If neither `ir_dataset_name` and `qrel_json` are provided.
    """

    if ir_dataset_name is None and qrel_json is None:
        raise ValueError(
            "One of ir_dataset_name or qrel_json must be provided."
        )

    if ir_dataset_name is not None and qrel_json is not None:
        raise ValueError(
            "Only one of ir_dataset_name or qrel_json can be provided."
        )

    if qrel_json is not None:
        qrels = load_qrels_from_json(qrel_json)
    else:
        qrels = load_qrels_from_ir_datasets(ir_dataset_name)

    retrieval_path = Path(retrieval_path)
    retrieval_pretty_path = retrieval_path.with_suffix(".txt")

    pretty_print_standard_format(
        retrieval_path, output_file=retrieval_pretty_path
    )
    run = load_standard_format_as_run(retrieval_path, score_key="score")

    calculate_metrics_to_file(
        run, qrels, output_folder=output_dir, metric_names=metric_names
    )

    print(f"Retrieval results saved to {retrieval_pretty_path}")
    print(f"Metrics saved to {output_dir}")


def do_retrieval_shared(
    retriever_cls,
    retriever_kwargs: Dict,
    output_dir: str,
    ir_dataset_name: Optional[str] = None,
    query_jsonl: Optional[str] = None,
    qrel_json: Optional[str] = None,
    query_id_key: str = "id",
    query_text_key: str = "text",
    top_k: int = 1000,
    include_content: bool = True,
    do_eval: bool = True,
    metric_names: Optional[List[str]] = None,
) -> None:
    """Does retrieval and optionally evaluation.

    Args:
        retriever_cls (BaseRetriever): The retriever class to use.
        retriever_kwargs (Dict): The keyword arguments to pass to the retriever.
        output_dir (str): Path to the output directory which will contain the
            retrieval results and optionally the evaluation results.
        ir_dataset_name (Optional[str], optional): If provided is used to
            get the queries used for retrieval and qrels used for evaluation.
            If None, then `query_jsonl` must be provided and `qrel_json` must
            be provided if `do_eval` is True. Defaults to None.
        query_jsonl (Optional[str], optional): If provided is used as the
            queries for retrieval. If None, then `ir_dataset_name` must
            be provided. Defaults to None.
        qrel_json (Optional[str], optional): If provided is used as the qrels
            for evaluation. If None, then `ir_dataset_name` must
            be provided. Defaults to None.
        query_id_key (str, optional): The key in `query_jsonl` for the
            query ID. Not used if `ir_dataset_name` is provided. Defaults to
            "id".
        query_text_key (str, optional): The key in `query_jsonl` for the
            query text. Not used if `ir_dataset_name` is provided. Defaults to
            "text".
        top_k (int, optional): The number of top items to retrieve. Defaults to
            1000.
        retriever_kwargs (Optional[Dict], optional): Additional keyword
            arguments to pass to the retriever. Defaults to None.
        include_content (bool, optional): Whether to include the content of the
            retrieved items in the output. Defaults to True.
        do_eval (bool, optional): Whether to do evaluation. Defaults to True.
        metric_names (Optional[List[str]], optional): A list of metrics to
            compute. These are passed to IR-Measures so should be compatible.
            If None, a default set of metrics is found. Defaults to None.
    Raises:
        ValueError: If both `query_jsonl` and `ir_dataset_name` are provided.
        ValueError: If `do_eval` is True and `ir_dataset_name` is None and
            `qrel_json` is None.
    """

    if query_jsonl is not None and ir_dataset_name is not None:
        raise ValueError(
            "Only one of query_jsonl and ir_dataset_name can be provided."
        )

    if query_jsonl is not None and do_eval and qrel_json is None:
        raise ValueError(
            "If do_eval is True and ir_dataset_name is None,"
            " qrel_json must be provided."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_file = output_dir / "retrieved_items.jsonl"
    metric_dir = output_dir / "metrics"

    retriever = retriever_cls(
        **retriever_kwargs
    )

    if query_jsonl is not None:
        retrieve_for_jsonl_queries(
            retriever=retriever,
            query_jsonl=query_jsonl,
            output_path=retrieval_file,
            top_k=top_k,
            include_content=include_content,
            include_type=include_content,
            query_id_key=query_id_key,
            query_text_key=query_text_key,
        )
    else:
        retrieve_for_ir_dataset_queries(
            retriever=retriever,
            ir_dataset_name=ir_dataset_name,
            output_path=retrieval_file,
            top_k=top_k,
            include_content=include_content,
            include_type=include_content,
            track_time=True,
        )

    if do_eval:
        do_eval_and_pretty_print(
            ir_dataset_name=ir_dataset_name,
            qrel_json=qrel_json,
            retrieval_path=retrieval_file,
            output_dir=metric_dir,
            metric_names=metric_names,
        )


def do_retrieval(
    model_name_or_path: str,
    encoded_item_path: str,
    output_dir: str,
    ir_dataset_name: Optional[str] = None,
    query_jsonl: Optional[str] = None,
    qrel_json: Optional[str] = None,
    query_id_key: str = "id",
    query_text_key: str = "text",
    dtype: str = "fp32",
    top_k: int = 1000,
    batch_size: int = 100_000,
    retriever_kwargs: Optional[Dict] = None,
    query_max_length: int = 64,
    include_content: bool = True,
    do_eval: bool = True,
    metric_names: Optional[List[str]] = None,
    ignore_same_id: bool = False,
) -> None:
    """Does retrieval and optionally evaluation.

    Args:
        model_name_or_path (str): Name or path to a HypencoderDualEncoder
            checkpoint.
        encoded_item_path (str): Path to the encoded items.
        output_dir (str): Path to the output directory which will contain the
            retrieval results and optionally the evaluation results.
        ir_dataset_name (Optional[str], optional): If provided is used to
            get the queries used for retrieval and qrels used for evaluation.
            If None, then `query_jsonl` must be provided and `qrel_json` must
            be provided if `do_eval` is True. Defaults to None.
        query_jsonl (Optional[str], optional): If provided is used as the
            queries for retrieval. If None, then `ir_dataset_name` must
            be provided. Defaults to None.
        qrel_json (Optional[str], optional): If provided is used as the qrels
            for evaluation. If None, then `ir_dataset_name` must
            be provided. Defaults to None.
        query_id_key (str, optional): The key in `query_jsonl` for the
            query ID. Not used if `ir_dataset_name` is provided. Defaults to
            "id".
        query_text_key (str, optional): The key in `query_jsonl` for the
            query text. Not used if `ir_dataset_name` is provided. Defaults to
            "text".
        dtype (str, optional): The dtype to use for the model and embedded
            items. Options are "fp16", "fp32", and "bf16". Defaults to "fp32".
        top_k (int, optional): The number of top items to retrieve. Defaults to
            1000.
        batch_size (int, optional): The batch size to use for retrieval.
            Defaults to 100,000.
        retriever_kwargs (Optional[Dict], optional): Additional keyword
            arguments to pass to the retriever. Defaults to None.
        query_max_length (int, optional): Maximum length of the query.
            Defaults to 64.
        include_content (bool, optional): Whether to include the content of the
            retrieved items in the output. Defaults to True.
        do_eval (bool, optional): Whether to do evaluation. Defaults to True.
        metric_names (Optional[List[str]], optional): A list of metrics to
            compute. These are passed to IR-Measures so should be compatible.
            If None, a default set of metrics is found. Defaults to None.
        ignore_same_id (bool, optional): Whether to ignore retrievals with the
            same ID as the query. This is only relevant for certain datasets.
            Defaults to False.

    Raises:
        ValueError: If both `query_jsonl` and `ir_dataset_name` are provided.
        ValueError: If `do_eval` is True and `ir_dataset_name` is None and
            `qrel_json` is None.
    """

    retriever_kwargs = retriever_kwargs if retriever_kwargs is not None else {}

    do_retrieval_shared(
        retriever_cls=HypencoderRetriever,
        retriever_kwargs=dict(
            model_name_or_path=model_name_or_path,
            encoded_item_path=encoded_item_path,
            dtype=dtype,
            batch_size=batch_size,
            query_max_length=query_max_length,
            ignore_same_id=ignore_same_id,
            **retriever_kwargs,
        ),
        output_dir=output_dir,
        ir_dataset_name=ir_dataset_name,
        query_jsonl=query_jsonl,
        qrel_json=qrel_json,
        query_id_key=query_id_key,
        query_text_key=query_text_key,
        top_k=top_k,
        include_content=include_content,
        do_eval=do_eval,
        metric_names=metric_names,
    )


if __name__ == "__main__":
    fire.Fire(do_retrieval)
