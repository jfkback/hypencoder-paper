import json
from collections import defaultdict
from numbers import Number
from typing import Dict

import ir_datasets


def load_qrels_from_ir_datasets(
    dataset_name: str,
    binarize: bool = False,
    binarize_threshold: int = 1,
) -> Dict[str, Dict[str, Number]]:
    """
    Load the qrels from ir_datasets.

    Args:
        dataset_name (str): The dataset name to use.

    Returns:
        Dict[str, Dict[str, Number]]: The qrels.
    """
    dataset = ir_datasets.load(dataset_name)
    qrels = defaultdict(dict)

    for qrel in dataset.qrels_iter():
        relevance = int(qrel.relevance)

        if binarize:
            relevance = relevance if relevance >= binarize_threshold else 0

        qrels[str(qrel.query_id)][str(qrel.doc_id)] = relevance

    return qrels


def load_qrels_from_json(
    input_json: str,
) -> Dict[str, Dict[str, Number]]:
    """
    Load the qrels from a json file.

    Args:
        input_json (str): The input json file.

    Returns:
        Dict[str, Dict[str, Number]]: The qrels.
    """

    with open(input_json, "r") as f:
        qrels = json.load(f)

    return qrels
