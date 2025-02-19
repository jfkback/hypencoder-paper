## Overview
This directory contains the files to easily encode and retrieve items using Hypencoder models. As well as providing evaluation metrics.

### Encoding and Retrieving
If the queries and documents you want to retrieve exist as a dataset in the IR Dataset library no additional work is needed to encode and retrieve from the dataset. If the data is not a part of this library you will need two JSONL files for the documents and queries. These must have the format:
```
{"<id_key>": "afei1243", "<text_key>": "This is some text"}
...
```
where `<id_key>` and `<text_key>` can be any string and do not have to be the same for the document and query file.

#### Encoding
```
export ENCODING_PATH="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
python hypencoder_cb/inference/encode.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--output_path=$ENCODING_PATH \
--jsonl_path=path/to/documents.jsonl \
--item_id_key=<id_key> \
--item_text_key=<text_key>
```
For all the arguments and information on using IR Datasets type:
`python hypencoder_cb/inference/encode.py --help`.

#### Retrieve
The values of `ENCODING_PATH` and `MODEL_NAME_OR_PATH` should be the same as
those used in the encoding step.
```
export ENCODING_PATH="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export RETRIEVAL_DIR="..."
python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--query_jsonl=path/to/queries.jsonl \
--do_eval=False \
--query_id_key=<id_key> \
--query_text_key=<text_key> \
--query_max_length=64 \
--top_k=1000
```
For all the arguments and information on using IR Datasets type:
`python hypencoder_cb/inference/retrieve.py --help`.

#### Evaluation
Evaluation is done automatically when `hypencoder_cb/inference/retrieve.py` is called so long as `--do_eval=True`. If you are not using an IR Dataset you will need to provide the qrels with the argument `--qrel_json`. The qrels JSON should be in the format:
```
{
    "qid1": {
        "pid8": relevance_value (float),
        "pid65": relevance_value (float),
        ...
    }.
    "qid2": {
        ...
    },
    ...
}
```

#### Approximate Retrieval
##### Getting a Item Neighbor Graph
Approximate retrieval requires an item-to-item graph. To get this graph use the following command:
```
export ENCODING_PATH="..."
export ITEM_NEIGHBOR_GRAPH="..."
python hypencoder_cb/inference/neighbor_graph.py \
--encoded_item_path=$ENCODING_PATH \
--output_path=$ITEM_NEIGHBOR_GRAPH \
--batch_size=100 \
--top_k=100 \
--device=cuda
```

##### Doing approximate retrieval
The values of `ENCODING_PATH` and `MODEL_NAME_OR_PATH` should be the same as
those used in the encoding step. Similarly `ENCODING_PATH` should be the same as the one used to construct the neighbor graph.
```
export ENCODING_PATH="..."
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ITEM_NEIGHBOR_GRAPH="..."
export RETRIEVAL_DIR="..."
python hypencoder_cb/inference/approx_retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--item_neighbors_path=$ITEM_NEIGHBOR_GRAPH \
--output_dir=$RETRIEVAL_DIR \
--query_jsonl=path/to/queries.jsonl \
--do_eval=False \
--query_id_key=<id_key> \
--query_text_key=<text_key> \
--query_max_length=64 \
--top_k=1000
```