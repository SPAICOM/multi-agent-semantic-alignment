# Federated Latent Space Alignment for Multi-user Semantic Communications

## Simulations

### Accuracy Vs Compression Factor

```bash
uv run scripts/train_linear.py communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 simulation=compr_fact -m
```

```bash
uv run scripts/train_baseline.py communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 base_station.strategy=First-K simulation=compr_fact -m
```

```bash
uv run scripts/train_baseline.py communication.channel_usage=1,2,3,4,5,10 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 base_station.strategy=Top-K simulation=compr_fact -m
```

### Accuracy Vs SNR

```bash
uv run scripts/train_linear.py communication.channel_usage=1,4,8 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 simulation=snr -m
```

```bash
uv run scripts/train_baseline.py communication.channel_usage=1,4,8 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 base_station.strategy=First-K simulation=snr -m
```

```bash
uv run scripts/train_baseline.py communication.channel_usage=2,4 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 base_station.strategy=Top-K simulation=snr -m
```

### Heterogenous Vs Homogeneous

```bash
uv run scripts/train_linear.py --config-name=heterogeneous communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 -m
```

```bash
uv run scripts/train_linear.py --config-name=homogeneous communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 -m
```

### Classifiers

```bash
uv run scripts/train_classifier.py rx_enc=vit_small_patch16_224,vit_small_patch32_224,vit_base_patch16_224,vit_base_patch32_clip_224,rexnet_100,mobilenetv3_small_075,mobilenetv3_large_100,mobilenetv3_small_100,efficientvit_m5.r224_in1k,levit_128s.fb_dist_in1k,vit_tiny_patch16_224 seed=27,42,100,123,144,200 -m
```

## Dependencies  

### Using `pip` package manager  

It is highly recommended to create a Python virtual environment before installing dependencies. In a terminal, navigate to the root folder and run:  

```bash
python -m venv <venv_name>
```

Activate the environment:  

- On macOS/Linux:  

  ```bash
  source <venv_name>/bin/activate
  ```

- On Windows:  

  ```bash
  <venv_name>\Scripts\activate
  ```

Once the virtual environment is active, install the dependencies:  

```bash
pip install -r requirements.txt
```

You're ready to go! 🚀  

### Using `uv` package manager (Highly Recommended)  

[`uv`](https://github.com/astral-sh/uv) is a modern Python package manager that is significantly faster than `pip`.  

#### Install `uv`  

To install `uv`, follow the instructions from the [official installation guide](https://github.com/astral-sh/uv#installation).  

#### Set up the environment and install dependencies  

Simply run a script with:

```bash
uv run path/to/script.py
```

Or Run the following command in the root folder:  

```bash
uv sync
```

This will automatically create a virtual environment (if none exists) and install all dependencies.  

You're ready to go! 🚀  

## Authors

- [Giuseppe Di Poce](https://github.com/giuseppedipoce)
- [Mario Edoardo Pandolfo](https://github.com/JRhin)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?hl=en&user=VZYvspQAAAAJ)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
