import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from tqdm import tqdm

from hypencoder_cb.utils.iterator_utils import BackgroundGenerator, batchify
from hypencoder_cb.utils.jsonl_utils import JsonlReader, JsonlWriter


@dataclass
class BaseEncodedRepresentation:
    representation: Any


@dataclass
class VectorEncodedRepresentation(BaseEncodedRepresentation):
    representation: np.ndarray


@dataclass
class Item:
    text: Optional[str] = None
    id: Optional[str] = None
    score: Optional[float] = None
    type: Optional[str] = None
    other: Dict = field(default_factory=dict)


@dataclass
class BaseQuery:
    id: Optional[str] = None


@dataclass
class EmbeddingQuery(BaseQuery):
    representation: Optional[np.ndarray] = None


@dataclass
class TextQuery(BaseQuery):
    text: Optional[str] = None
    other: Dict = field(default_factory=dict)


class EncodedItem(BaseDoc):
    text: str
    representation: NdArray
    id: Optional[str] = None


def items_from_ir_dataset(
    ir_dataset_name: str,
) -> Iterable[Item]:
    import ir_datasets

    dataset = ir_datasets.load(ir_dataset_name)

    for doc in dataset.docs_iter():
        text = ""
        if hasattr(doc, "title"):
            text += f"{doc.title} "
        text += doc.text

        yield Item(text=text.strip(), id=doc.doc_id)


def items_from_jsonl(
    jsonl_path: str,
    text_key: str = "item_text",
    id_key: str = "item_id",
) -> Iterable[Item]:
    with JsonlReader(jsonl_path) as reader:
        for line in reader:
            yield Item(text=line[text_key], id=line[id_key])


class BaseEncoder:
    encoding_type: BaseEncodedRepresentation = BaseEncodedRepresentation

    def encode(self, text: str) -> BaseEncodedRepresentation:
        raise NotImplementedError()

    def batch_encode(
        self, texts: List[str]
    ) -> List[BaseEncodedRepresentation]:
        raise NotImplementedError()


def encode_items(
    encoder: BaseEncoder, items: Iterable[Item], batch_size: int = 32
) -> Iterable[EncodedItem]:
    with tqdm() as pbar:
        for batch in BackgroundGenerator(batchify(items, batch_size), 10):
            output = encoder.batch_encode([item.text for item in batch])

            for item, encoded_rep in zip(
                batch,
                output.representation.cpu().to(dtype=torch.float32).numpy(),
            ):
                pbar.update(1)

                yield EncodedItem(
                    text=item.text,
                    representation=encoded_rep,
                    id=item.id,
                )


def encode_items_to_disk(
    encoder: BaseEncoder,
    items: Iterable[Item],
    output_path: str,
    batch_size: int = 32,
) -> None:
    encoded_iter = encode_items(encoder, items, batch_size=batch_size)
    DocList[EncodedItem].push_stream(
        encoded_iter,
        f"file://{output_path}",
    )


def encode_ir_dataset_items_to_disk(
    encoder: BaseEncoder,
    ir_dataset_name: str,
    output_path: str,
    batch_size: int = 32,
) -> None:
    items = items_from_ir_dataset(ir_dataset_name)
    encode_items_to_disk(encoder, items, output_path, batch_size=batch_size)


def encode_jsonl_items_to_disk(
    encoder: BaseEncoder,
    input_path: str,
    output_path: str,
    batch_size: int = 32,
    item_text_key: str = "item_text",
    item_id_key: str = "item_id",
) -> None:
    items = items_from_jsonl(
        input_path, text_key=item_text_key, id_key=item_id_key
    )
    encode_items_to_disk(encoder, items, output_path, batch_size=batch_size)


class BaseRetriever:
    implements_retrieve_text = False
    implements_retrieve = False

    def retrieve_text(
        self, query: TextQuery, top_k: Optional[int] = None
    ) -> List[Item]:
        raise NotImplementedError

    def retrieve(
        self, query: BaseQuery, top_k: Optional[int] = None
    ) -> List[Item]:
        raise NotImplementedError


def load_encoded_items_from_disk(
    encoded_items_path: str,
) -> Iterable[EncodedItem]:
    return DocList[EncodedItem].pull(
        f"file://{encoded_items_path}", show_progress=True
    )


def query_to_json(query: BaseQuery) -> Dict:
    output = {}

    if hasattr(query, "text") and query.text is not None:
        output["content"] = query.text

    if hasattr(query, "id") and query.id is not None:
        output["id"] = query.id

    return output


def item_to_json(
    item: Item, include_content: bool = True, include_type: bool = True
) -> Dict:
    output = {}

    if item.text is not None and include_content:
        output["content"] = item.text

    if item.id is not None:
        output["id"] = item.id

    if item.score is not None:
        output["score"] = item.score

    if item.type is not None and include_type:
        output["type"] = item.type

    return output


def query_items_to_jsonl(
    query_items: Iterable[Tuple[TextQuery, List[Item]]],
    output_path: str,
    item_to_jsonl_kwargs: Optional[Dict] = None,
    append: bool = False,
) -> None:

    if item_to_jsonl_kwargs is None:
        item_to_jsonl_kwargs = {}

    mode = "a" if append else "w"

    with JsonlWriter(output_path, mode=mode) as writer:
        for query, items in query_items:
            writer.write(
                {
                    "query": query_to_json(query),
                    "items": [
                        item_to_json(item, **item_to_jsonl_kwargs)
                        for item in items
                    ],
                }
            )


def retrieve_items(
    retriever: BaseRetriever,
    queries: Iterable[TextQuery],
    top_k: Optional[int] = None,
    track_time: bool = False,
    track_time_file: Optional[str] = None,
) -> Iterable[Tuple[TextQuery, List[Item]]]:
    if track_time:
        start_time = time.time()
        num_queries = 0

    for query in tqdm(queries):
        yield query, retriever.retrieve(query, top_k=top_k)

        if track_time:
            num_queries += 1

    if track_time:
        end_time = time.time()
        print(
            f"Retrieved {num_queries} queries in {end_time - start_time}"
            f" seconds. That's {num_queries / (end_time - start_time)} queries"
            " per second. Average time per query:"
            f" {(end_time - start_time) / num_queries}"
        )

        if track_time_file is not None:
            with open(track_time_file, "w") as f:
                json.dump(
                    {
                        "num_queries": num_queries,
                        "time": end_time - start_time,
                    },
                    f,
                )


def retrieve_for_ir_dataset_queries(
    retriever: BaseRetriever,
    ir_dataset_name: str,
    output_path: str,
    top_k: Optional[int] = None,
    include_type: bool = True,
    include_content: bool = True,
    append_output: bool = False,
    skip_queries: Optional[Set[str]] = None,
    track_time: bool = False,
    track_time_file: Optional[str] = None,
) -> None:
    import ir_datasets

    dataset = ir_datasets.load(ir_dataset_name)

    if skip_queries is None:
        skip_queries = set()

    queries = [
        TextQuery(id=query.query_id, text=query.text)
        for query in dataset.queries_iter()
        if query.query_id not in skip_queries
    ]

    query_items = retrieve_items(
        retriever,
        queries,
        top_k=top_k,
        track_time=track_time,
        track_time_file=track_time_file,
    )
    query_items_to_jsonl(
        query_items,
        output_path,
        item_to_jsonl_kwargs=dict(
            include_content=include_content, include_type=include_type
        ),
        append=append_output,
    )


def retrieve_for_jsonl_queries(
    retriever: BaseRetriever,
    query_jsonl: str,
    output_path: str,
    top_k: Optional[int] = None,
    include_type: bool = True,
    include_content: bool = True,
    query_id_key: str = "query_id",
    query_text_key: str = "query_text",
    max_p_converter: Optional[Callable[[List[Item]], List[Item]]] = None,
) -> None:
    with JsonlReader(query_jsonl) as reader:
        queries = [
            TextQuery(id=query[query_id_key], text=query[query_text_key])
            for query in reader
        ]

    query_items = retrieve_items(retriever, queries, top_k=top_k)

    query_items_to_jsonl(
        (
            (
                query,
                (
                    max_p_converter(items)
                    if max_p_converter is not None
                    else items
                ),
            )
            for query, items in query_items
        ),
        output_path,
        item_to_jsonl_kwargs=dict(
            include_content=include_content, include_type=include_type
        ),
    )
