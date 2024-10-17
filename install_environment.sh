#!/bin/bash

#install requirements
pip install -r requirements.txt

# install pytorch geometric dev from my fork
pip install --force-reinstall git+https://github.com/elilaird/pytorch_geometric.git@qm40-dataset
