#!/bin/bash

conda_init="/g/emcf/software/python/miniconda/etc/profile.d/conda.sh"

source $conda_init
conda init bash > /dev/null

conda activate xtomo
