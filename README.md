# N-Gram Graph

This repository contains the source code for the paper
> Shengchao Liu, Mehmet Furkan Demirel, Yingyu Liang. N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules. NeurIPS 2019 (Spotlight).

You can check the paper on  [NeurIPS proceedings](https://papers.nips.cc/paper/9054-n-gram-graph-simple-unsupervised-representation-for-graphs-with-applications-to-molecules) or [ArXiv](https://arxiv.org/abs/1806.09206).

### 1. Env Setup
Install Anaconda2-4.3.1 first, and below is an example on Linux.
```
wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
bash Anaconda2-4.3.1-Linux-x86_64.sh -b -p ./anaconda
export PATH=$PWD/anaconda/bin:$PATH
```

Then set up the env.
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

Below is the specification of all the datasets & tasks.

| Dataset | # of Tasks | Task Type |
| :---: | :---: | :---: |
| Delaney | 1 | Regression |
| Malaria | 1 | Regression |
| CEP | 1 | Regression |
| QM7 | 1 | Regression |
| QM8 | 12 | Regression |
| QM9 | 12 | Regression |
| Tox21 | 12 | Classification |
| Clintox | 2 | Classification |
| MUV | 17 | Classification |
| HIV | 1 | Classification |

### 3. Run Models

There are two `test.sh` scripts (under path `n_gram_graph/` and `n_gram_graph/embedding/`) for quick test on task `Delaney`.

#### 3.1 Run the Node-Level and Graph-Level Embedding

+ First specify the arguments.
```
cd n_gram_graph/embedding

export task=...
export running_index=...
```

+ Run the node-level embedding:
```
mkdir -p ./model_weight/"$task"/"$running_index"

python node_embedding.py \
--mode="$task" \
--running_index="$running_index"
```

+ Run the graph-level embedding:
```
mkdir -p ../../datasets/"$task"/"$running_index"

python graph_embedding.py \
--mode="$task" \
--running_index="$running_index"
```

Please check `run_embedding.sh` for detailed specifications.

#### 3.2 Run RF and XGB

+ First specify arguments.
```
cd n_gram_graph

export task=...
export model=...
export weight_file=...
export running_index=...
```

+ For classification tasks:
```
python main_classification.py \
--task="$task" \
--config_json_file=../config/"$model"/"$task".json \
--weight_file="$weight_file" \
--running_index="$running_index" \
--model="$model" 
```

+ For regression tasks:
```
python main_regression.py \
--task="$task" \
--config_json_file=../config/"$model"/"$task".json \
--weight_file="$weight_file" \
--running_index="$running_index" \
--model="$model" 
```

Please check `run_n_gram_classification.sh` and `run_n_gram_regression.sh` for detailed specifications.

### 4 Citation

```
@incollection{NIPS2019_9054,
    title = {N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules},
    author = {Liu, Shengchao and Demirel, Mehmet F and Liang, Yingyu},
    booktitle = {Advances in Neural Information Processing Systems 32},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {8464--8476},
    year = {2019},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/9054-n-gram-graph-simple-unsupervised-representation-for-graphs-with-applications-to-molecules.pdf}
}

```
