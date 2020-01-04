#!/usr/bin/env bash

task_list=(delaney malaria cep qm7 E1-CC2 E2-CC2 f1-CC2 f2-CC2 E1-PBE0 E2-PBE0 f1-PBE0 f2-PBE0 E1-CAM E2-CAM f1-CAM f2-CAM mu alpha homo lumo gap r2 zpve cv u0 u298 h298 g298)
running_index_list=(0 1 2 3 4)
model_list=(n_gram_rf n_gram_xgb)

for task in "${task_list[@]}"; do
    for model in "${model_list[@]}"; do
        for running_index in "${running_index_list[@]}"; do
            mkdir -p ../output/"$model"/"$running_index"

            python main_regression.py \
            --task="$task" \
            --config_json_file=../config/"$model"/"$task".json \
            --weight_file=temp.pt \
            --running_index="$running_index" \
            --model="$model" > ../output/"$model"/"$running_index"/"$task".out
        done
    done
done