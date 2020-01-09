#!/usr/bin/env bash

task_list=(qm7 qm8 qm9 delaney malaria cep
tox21/NR-AR tox21/NR-AR-LBD
tox21/NR-AhR tox21/NR-Aromatase
tox21/NR-ER tox21/NR-ER-LBD
tox21/NR-PPAR-gamma tox21/SR-ARE
tox21/SR-ATAD5 tox21/SR-HSE
tox21/SR-MMP tox21/SR-p53
muv/MUV-466 muv/MUV-548 muv/MUV-600 muv/MUV-644
muv/MUV-652 muv/MUV-689 muv/MUV-692 muv/MUV-712
muv/MUV-713 muv/MUV-733 muv/MUV-737 muv/MUV-810
muv/MUV-832 muv/MUV-846 muv/MUV-852 muv/MUV-858
muv/MUV-859 hiv clintox/CT_TOX clintox/FDA_APPROVED)
running_index_list=(0 1 2 3 4)

for task in "${task_list[@]}"; do
    for running_index in "${running_index_list[@]}"; do
        mkdir -p ./output/"$task"/"$running_index"
        mkdir -p ./model_weight/"$task"/"$running_index"

        python node_embedding.py \
        --mode="$task" \
        --running_index="$running_index" > ./output/"$task"/"$running_index"/node_embedding.out

        mkdir -p ../../datasets/"$task"/"$running_index"

        python graph_embedding.py \
        --mode="$task" \
        --running_index="$running_index" > ./output/"$task"/"$running_index"/graph_embedding.out
    done
done