# In-Domain Results
## Set general options
```
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="/tmp/encoded_items/..."
export RETRIEVAL_DIR="/tmp/retrievals/..."
```

## Set dataset
Pick the dataset you want to use from the list below:
```
export IR_DATASET_NAME="msmarco-passage/trec-dl-2020/judged"
export IR_DATASET_NAME="msmarco-passage/trec-dl-2019/judged"
export IR_DATASET_NAME="msmarco-passage/dev/small"
```

## Encode
```
python hypencoder_cb/inference/encode.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--output_path=$ENCODING_PATH \
--ir_dataset_name=$IR_DATASET_NAME
```

## Retrieve
```
python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=64
```


# Out-of-Domain Results
## Set general options
```
export MODEL_NAME_OR_PATH="jfkback/hypencoder.6_layer"
export ENCODING_PATH="/tmp/encoded_items/..."
export RETRIEVAL_DIR="/tmp/retrievals/..."
```

## Set dataset
Pick the dataset you want to use from the list below:
```
export IR_DATASET_NAME="beir/fiqa/test"
export IR_DATASET_NAME="beir/trec-covid"
export IR_DATASET_NAME="beir/nfcorpus/test"
export IR_DATASET_NAME="beir/dbpedia-entity/test"
export IR_DATASET_NAME="beir/webis-touche2020/v2"
```

## Encode
```
python hypencoder_cb/inference/encode.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--output_path=$ENCODING_PATH \
--ir_dataset_name=$IR_DATASET_NAME
```

## Retrieve
```
python hypencoder_cb/inference/retrieve.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--encoded_item_path=$ENCODING_PATH \
--output_dir=$RETRIEVAL_DIR \
--ir_dataset_name=$IR_DATASET_NAME \
--query_max_length=512
```