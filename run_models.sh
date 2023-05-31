#!/bin/bash

sbatch -J multiclass_cnn_vgg19 job_cnn.slurm multiclass_cnn_vgg19 
sbatch -J multiclass_cnn_xception job_cnn.slurm multiclass_cnn_xception 
sbatch -J multiclass_vit job_vit.slurm multiclass_vit 

sbatch -J multiclass_cnn_vgg19_header job_cnn.slurm multiclass_cnn_vgg19_header 
sbatch -J multiclass_cnn_xception_header job_cnn.slurm multiclass_cnn_xception_header 
sbatch -J multiclass_vit_header job_vit.slurm multiclass_vit_header

sbatch -J binary_cnn_vgg19 job_cnn.slurm binary_cnn_vgg19 
sbatch -J binary_cnn_xception job_cnn.slurm binary_cnn_xception 
sbatch -J binary_vit job_vit.slurm binary_vit 

sbatch -J binary_cnn_vgg19_header job_cnn.slurm binary_cnn_vgg19_header 
sbatch -J binary_cnn_xception_header job_cnn.slurm binary_cnn_xception_header 
sbatch -J binary_vit_header job_vit.slurm binary_vit_header 