#!/usr/bin/env bash

# regression datasets
python prepare_delaney.py > prepare_delaney.out
python prepare_malaria.py > prepare_malaria.out
python prepare_cep.py > prepare_cep.out
python prepare_qm7.py > prepare_qm7.out
python prepare_qm8.py > prepare_qm8.out
python prepare_qm9.py > prepare_qm9.out

# classification datasets
python prepare_tox21.py > prepare_tox21.out
python prepare_clintox.py > prepare_clintox.out
python prepare_muv.py > prepare_muv.out
python prepare_hiv.py > prepare_hiv.out
