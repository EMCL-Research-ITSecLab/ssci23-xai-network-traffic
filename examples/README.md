# Prepare Data set for Training

Download the USTC- data set https://github.com/yungshenglu/USTC-TFC2016

Split the PCAP with PcapPlusPlus

```sh
python convert_pcaps.py
```

Split data to train, test, and validation

```sh
python split_data.py ratio -r /home/smachmeier/data/multiclass-flow-minp2-dim16-cols8-ALL-HEADER -w /home/smachmeier/data/multiclass-flow-minp2-dim16-cols8-ALL-HEADER-split-ratio
```

Convert to images

```sh
pip install fip

fip extract flow-tiled-fixed -t 32 -r {PATH_PCAPS} -w {PATH_IMAGES} --min-packets 3 --dim 16 --cols 8 [--preprocess HEADER]
```
