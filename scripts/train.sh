#!/bin/bash

dataset=data/nerf_synthetic
scene=lego

# python train.py -s $dataset/$scene -m output/$scene

# python render.py -s $dataset/$scene -m output/$scene --unbounded --skip_test --skip_train --mesh_res 1024
python render.py -s $dataset/$scene -m output/$scene --skip_test --skip_train --mesh_res 1024