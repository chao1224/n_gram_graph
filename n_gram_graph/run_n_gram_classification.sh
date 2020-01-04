#!/usr/bin/env bash

task_list=(hiv MUV-466 MUV-548 MUV-600 MUV-644 MUV-652 MUV-689 MUV-692 MUV-712 MUV-713 MUV-733 MUV-737 MUV-810 MUV-832 MUV-846 MUV-852 MUV-858 MUV-859 NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD NR-PPAR-gamma SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53 CT_TOX FDA_APPROVED)
running_index_list=(0 1 2 3 4)
model_list=(n_gram_rf n_gram_xgb)

for task in "${task_list[@]}"; do
    for model in "${model_list[@]}"; do
        for running_index in "${running_index_list[@]}"; do
            mkdir -p ../output/"$model"/"$running_index"

            python main_classification.py \
            --task="$task" \
            --config_json_file=../config/"$model"/"$task".json \
            --weight_file=temp.pt \
            --running_index="$running_index" \
            --model="$model" > ../output/"$model"/"$running_index"/"$task".out
        done
    done
done