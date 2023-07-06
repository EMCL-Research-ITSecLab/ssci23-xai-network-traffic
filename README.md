# SSCI - Explainable artificial intelligence for improving a session-based malware traffic classification with deep learning


In network security, applying deep learning methods to detect network traffic anomalies has achieved great results with various network traffic representations.
A possible representation is the transformation of raw network communication to images to extract valuable information from the unmanageable amount of network traffic by applying representation learning.
However, since deep learning models can result in black boxes for users, it is interesting to understand what valuable information is learned from network communication converted into images.
This paper elaborates on that question using explainable artificial intelligence (XAI) methods to identify network packets that most influence the prediction and verify that packets in a malware communication containing malicious payloads have a higher influence on the prediction.
We inspect the Grad-CAM and visualize the Integrated Gradients of the Xception and VGG-19 model and investigate the attention heat maps of our Vision Transformer (ViT) model.
In addition, we present a novel transformation of sessions to a new image representation to expand the informativeness of network communication.
For multiclass classification, our best model Xception achieves an accuracy of $97.95\%$, whereas, for binary classification, Xception and VGG-19 achieve well above $99.50\%$.
Our ViT model achieves a significantly lower performance with $95.86\%$ for multiclass and $99.36\%$ for binary classification.
In particular, computing centers could benefit by examining their inbound and outbound traffic to detect malicious behaviors ahead of time.

All trained models and data sets are available on [heiBOX](https://heibox.uni-heidelberg.de/d/163d02d93b41401783bc/)

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
