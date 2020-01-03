# N-Gram Graph

This repository contains the source code for the paper
> Shengchao Liu, Mehmet Furkan Demirel, Yingyu Liang. [N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules
](https://arxiv.org/abs/1806.09206). NeurIPS 2019 (Spotlight).

### 1. Env Setup
```
conda env create -f gpu_env.yml
source activate n_gram_project
pip install --user -e .
```

### 2. Data Preparation
```
cd datasets
bash download_data.sh
bash data_preprocess.sh
```

### 3. Run Models

This shows the test script for task `Delaney`.

#### 3.1 Run the Node-Level and Graph-Level Embedding

```
cd n_gram_graph/node2vec
bash test.sh
```

#### 3.2 Run the RF and XGB

```
cd n_gram_graph
bash test.sh
```

### Citation

```
@article{liu2019ngg,
    title={N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules},
    author={Liu, Shengchao and Demirel, Mehmet Furkan and Liang, Yingyu},
    booktitle={Neural Information Processing Systems},
    year={2019}
}
```