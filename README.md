# Hypencoder
Official Repository for "Hypencoder: Hypernetworks for Information Retrieval". Code coming soon.

![main_image](./imgs/main_figure.jpg)

<h4 align="center">
    <p>
        <a href=#installation>Installation</a> |
        <a href=#quick-start>Quick Start</a> |
        <a href="https://arxiv.org/pdf/2502.05364">Paper</a> |
        <a href=#models>Models</a> |
        <a href="#cite">Citation</a>
    <p>
</h4>

## Installation
### Copy the Repo
```
gh repo clone jfkback/hypencoder-paper
```

### Install locally with pip
```
pip install -e /hypencoder-paper
```

### Required Libraries
The core libraries required are:
- `torch`
- `transformers`
with just the core libraries you can use Hypencoder to create q-nets and
document embeddings.

To use the code for encoding and retrieval the following additional libraries
are required:
- `fire`
- `tqdm`
- `ir_datasets`
- `jsonlines`
- `docarray`
- `numpy`
- `ir_measures`


## Quick Start


## Models
| Huggingface Repo | Number of Layers |
|------------------|------------------|
| [jfkback/hypencoder.2_layer](https://huggingface.co/jfkback/hypencoder.2_layer) |          2        |
| [jfkback/hypencoder.4_layer](https://huggingface.co/jfkback/hypencoder.4_layer) |          4        |
| [jfkback/hypencoder.6_layer](https://huggingface.co/jfkback/hypencoder.6_layer) |          6        |
| [jfkback/hypencoder.8_layer](https://huggingface.co/jfkback/hypencoder.8_layer) |          8        |


## Citation
```
@misc{killingback2025hypencoderhypernetworksinformationretrieval,
      title={Hypencoder: Hypernetworks for Information Retrieval},
      author={Julian Killingback and Hansi Zeng and Hamed Zamani},
      year={2025},
      eprint={2502.05364},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2502.05364},
}
```