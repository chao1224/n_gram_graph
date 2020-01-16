#!/usr/bin/env bash

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/muv.csv.gz
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/HIV.csv
mv HIV.csv hiv.csv

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/delaney-processed.csv
wget -O malaria-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv
wget -O cep-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-02-cep-pce/cep-processed.csv

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.csv

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm8.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb8.tar.gz
tar -xzvf gdb8.tar.gz
rm gdb8.tar.gz

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
tar -xzvf gdb9.tar.gz
mv gdb9.sdf qm9.sdf
mv gdb9.sdf.csv qm9.sdf.csv
rm gdb9.tar.gz
