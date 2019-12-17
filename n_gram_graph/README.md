# N-Gram Graph

## Baselines

+ Morgan FPs + Random Forest
+ Morgan FPs + XGBoost

## N-Gram Graph

+ N-Gram Graph + Random Forest
+ N-Gram Graph + XGBoost

```
conda env create -f gpu_env.yml
pip install --user -e .

cd n_gram_graph/node2vec
# see run.sh for an example
python node_embedding.py --..
python graph_embedding.py --...

cd n_gram_graph
python baselines_classification.py --...
python baselines_regression.py --...
```

