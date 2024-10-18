#!/bin/bash

#install requirements
# Check if the "use-nvidia" argument is passed
if [ "$1" = "use-nvidia" ]; then
    echo "Using NVIDIA container"
else
    pip install -r requirements.txt
fi

# install pytorch geometric dev from my fork
pip install --force-reinstall git+https://github.com/elilaird/pytorch_geometric.git@qm40-dataset
