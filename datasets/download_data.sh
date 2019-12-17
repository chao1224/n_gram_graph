#!/usr/bin/env bash

#wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.mat
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb7.tar.gz
tar -xzvf gdb7.tar.gz
mv gdb7.sdf qm7.sdf

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm8.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb8.tar.gz
tar -xzvf gdb8.tar.gz


wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv
wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
tar -xzvf gdb9.tar.gz
mv gdb9.sdf qm9.sdf
mv gdb9.sdf.csv qm9.sdf.csv


rm gdb7.tar.gz
rm gdb8.tar.gz
rm gdb9.tar.gz

wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/full_smiles_labels.csv
mv full_smiles_labels.csv pdbbind.csv

#wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/featurized_datasets/full_grid.tar.gz
#tar -zxvf full_grid.tar.gz
#wget http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/pdbbind_v2015.tar.gz
