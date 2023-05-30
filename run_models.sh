#!/bin/bash
sbatch job_cnn.slurm multiclass_cnn_vgg19 -o multiclass_cnn_vgg19.out
sbatch job_cnn.slurm multiclass_cnn_xception -o multiclass_cnn_xception.out
sbatch job_vit.slurm multiclass_vit -o multiclass_vit