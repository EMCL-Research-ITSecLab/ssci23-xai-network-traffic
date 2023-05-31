#!/bin/bash

sbatch job_cnn.slurm multiclass_cnn_vgg19 -J multiclass_cnn_vgg19
sbatch job_cnn.slurm multiclass_cnn_xception -J multiclass_cnn_xception
sbatch job_vit.slurm multiclass_vit -J multiclass_vit

sbatch job_cnn.slurm multiclass_cnn_vgg19_header -J multiclass_cnn_vgg19_header
sbatch job_cnn.slurm multiclass_cnn_xception_header -J multiclass_cnn_xception_header
sbatch job_vit.slurm multiclass_vit_header -J multiclass_vit_header

sbatch job_cnn.slurm binary_cnn_vgg19 -J binary_cnn_vgg19
sbatch job_cnn.slurm binary_cnn_xception -J binary_cnn_xception
sbatch job_vit.slurm binary_vit -J binary_vit

sbatch job_cnn.slurm binary_cnn_vgg19_header -J binary_cnn_vgg19_header
sbatch job_cnn.slurm binary_cnn_xception_header -J binary_cnn_xception_header
sbatch job_vit.slurm binary_vit_header -J binary_vit_header