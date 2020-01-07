# N-Gram Graph

This repository contains the source code for the paper
> Shengchao Liu, Mehmet Furkan Demirel, Yingyu Liang. [N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules
](https://arxiv.org/abs/1806.09206). NeurIPS 2019 (Spotlight).

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

### 3. Run Models

The cmds below include the demo scripts for running task `Delaney`.

#### 3.1 Run the Node-Level and Graph-Level Embedding

```
cd n_gram_graph/embedding
bash test.sh
```

#### 3.2 Run the RF and XGB

```
cd n_gram_graph
bash test.sh
```

### Citation

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
