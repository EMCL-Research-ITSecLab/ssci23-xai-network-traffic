# SSCI - Session-based Image Traffic Classification

## Getting Started

Set up Conda environment:

```sh
conda create --name ssci python=3.10
```

Install `requirements.txt`

```sh
pip install -r requirements.txt
```

Start training:

```sh
python train_binary.py
```

## Troubleshooting

For any issues related to CUDA, check out [Install TensorFlow with pip](https://www.tensorflow.org/install/pip) guide.

```sh
for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done # https://github.com/tensorflow/tensorflow/issues/42738


export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
```

Other related [issue](https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path).