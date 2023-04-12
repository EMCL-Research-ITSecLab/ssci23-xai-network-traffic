# SSCI - Flow-based Image Traffic Classification

## Getting Started

Set up Conda environment:

```sh
conda create --name ssci python=3.10
```

Install `requirements.txt`

```sh
pip install -r requirements.txt
```

Convert images:

```sh
pip install fip

fip extract flow-tiled-fixed -t 32 -r {PATH_PCAPS} -w {PATH_IMAGES} --min-packets 3 --dim 16 --cols 8
```

Start training:

```sh
python train_binary.py
```

## Troubleshooting

For any issues related to CUDA, check out [Install TensorFlow with pip](https://www.tensorflow.org/install/pip) guide.

```sh
for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```

Other related [issue](https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path).