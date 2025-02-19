import random
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer


def positive_filter_factory(
    type: str, **kwargs
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    if type == "type":
        return lambda items: [
            item
            for item in items
            if item.get("type", "") == kwargs["positive_type"]
        ]
    elif type == "first":
        return lambda items: items[:1]
    elif type == "score_above":
        return lambda items: [
            item
            for item in items
            if item.get(kwargs.get("score_key", "score"), 0)
            > kwargs.get("score_threshold", 0)
        ]


def sampler_factory(
    type: str, num_samples, **kwargs
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    if type == "random":
        return lambda items: random.sample(items, k=num_samples)
    elif type == "all":
        return lambda items: items


class GeneralDualEncoderCollator:

    def __init__(
        self,
        tokenizer,
        num_negatives_to_sample: int,
        positive_filter: Union[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]], str
        ] = "type",
        positive_sampler: Union[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]], str
        ] = "random",
        negative_sampler: Union[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]], str
        ] = "random",
        num_positives_to_sample: int = 1,
        label_key: Optional[str] = "score",
        positive_filter_kwargs: Optional[Dict[str, Any]] = None,
        positive_sampler_kwargs: Optional[Dict[str, Any]] = None,
        negative_sampler_kwargs: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
        query_padding_mode: str = "longest",
        query_max_length: Optional[int] = None,
        modify_query: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        query_pad_kwargs: Optional[Dict[str, Any]] = None,
        item_pad_kwargs: Optional[Dict[str, Any]] = None,
    ):
        random.seed(random_seed)

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.num_negatives_to_sample = num_negatives_to_sample
        self.num_positives_to_sample = num_positives_to_sample

        if positive_filter_kwargs is None:
            positive_filter_kwargs = {}
        if positive_sampler_kwargs is None:
            positive_sampler_kwargs = {}
        if negative_sampler_kwargs is None:
            negative_sampler_kwargs = {}
        if query_pad_kwargs is None:
            query_pad_kwargs = {}
        if item_pad_kwargs is None:
            item_pad_kwargs = {}

        if isinstance(positive_filter, str):
            positive_filter = positive_filter_factory(
                positive_filter, **positive_filter_kwargs
            )
        if isinstance(positive_sampler, str):
            positive_sampler = sampler_factory(
                positive_sampler,
                num_positives_to_sample,
                **positive_sampler_kwargs,
            )
        if isinstance(negative_sampler, str):
            negative_sampler = sampler_factory(
                negative_sampler,
                num_negatives_to_sample,
                **negative_sampler_kwargs,
            )

        self.positive_filter = positive_filter
        self.positive_sampler = positive_sampler
        self.negative_sampler = negative_sampler

        self.label_key = label_key
        self.query_padding_mode = query_padding_mode
        self.query_max_length = query_max_length
        self.modify_query = modify_query
        self.query_pad_kwargs = query_pad_kwargs
        self.item_pad_kwargs = item_pad_kwargs

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        queries = [
            {"input_ids": f["query"]["tokenized_content"]} for f in features
        ]

        items, labels = [], []

        for i, feature in enumerate(features):
            positive_items = self.positive_filter(feature["items"])

            if len(positive_items) < self.num_positives_to_sample:
                raise ValueError(
                    f"Positive items less than num_positives_to_sample: {len(positive_items)}"
                )

            selected_items = self.positive_sampler(positive_items)

            assert len(selected_items) == self.num_positives_to_sample

            negative_items = [
                item for item in feature["items"] if item not in positive_items
            ]

            if len(negative_items) < self.num_negatives_to_sample:
                raise ValueError(
                    f"Negative items less than num_negatives_to_sample: {len(negative_items)}"
                )

            selected_items += self.negative_sampler(negative_items)

            assert (
                len(selected_items)
                == self.num_negatives_to_sample + self.num_positives_to_sample
            )

            items.extend(
                [
                    {"input_ids": item["tokenized_content"]}
                    for item in selected_items
                ]
            )

            if self.label_key is not None:
                labels.append(
                    [item[self.label_key] for item in selected_items]
                )

        query_inputs = self.tokenizer.pad(
            queries,
            padding=self.query_padding_mode,
            max_length=self.query_max_length,
            return_tensors="pt",
            **self.query_pad_kwargs,
        )

        if self.modify_query is not None:
            query_inputs = self.modify_query(query_inputs)

        item_inputs = self.tokenizer.pad(
            items,
            padding="longest",
            return_tensors="pt",
            **self.item_pad_kwargs,
        )

        if labels == []:
            labels = None
        else:
            labels = torch.tensor(labels)

        return {
            "query_input_ids": query_inputs["input_ids"],
            "query_attention_mask": query_inputs["attention_mask"],
            "passage_input_ids": item_inputs["input_ids"],
            "passage_attention_mask": item_inputs["attention_mask"],
            "labels": labels,
        }
