#!/bin/bash

sbatch job_cnn.slurm multiclass_cnn_vgg19 -o multiclass_cnn_vgg19.out
sbatch job_cnn.slurm multiclass_cnn_xception -o multiclass_cnn_xception.out
sbatch job_vit.slurm multiclass_vit -o multiclass_vit

sbatch job_cnn.slurm multiclass_cnn_vgg19_header -o multiclass_cnn_vgg19_header.out
sbatch job_cnn.slurm multiclass_cnn_xception_header -o multiclass_cnn_xception_header.out
sbatch job_vit.slurm multiclass_vit_header -o multiclass_vit_header.out

sbatch job_cnn.slurm binary_cnn_vgg19 -o binary_cnn_vgg19.out
sbatch job_cnn.slurm binary_cnn_xception -o binary_cnn_xception.out
sbatch job_vit.slurm binary_vit -o binary_vit

sbatch job_cnn.slurm binary_cnn_vgg19_header -o binary_cnn_vgg19_header.out
sbatch job_cnn.slurm binary_cnn_xception_header -o binary_cnn_xception_header.out
sbatch job_vit.slurm binary_vit_header -o binary_vit_header.out