# Federated Latent Space Alignment for Multi-user Semantic Communications

> [!TIP]
> Semantic communications focus on understanding the meaning behind transmitted data, ensuring effective task execution and seamless information exchange. However, when AI-native devices employ different internal representations (e.g., latent spaces), semantic mismatches can arise, hindering mutual comprehension. This paper introduces a novel approach to mitigating latent space misalignment in multi-agent AI-native semantic communications. In a downlink scenario, we consider an access point (AP) communicating with multiple users to accomplish a specific AI-driven task. Our method implements a protocol that shares semantic encoder at the AP and local semantic equalizers at user devices, fostering mutual understanding and task-oriented communication while considering power and complexity constraints. To achieve this, we employ a federated optimization for the decentralized training of semantic encoders and equalizers. Numerical results validate the proposed approach in goal-oriented semantic communication, revealing key trade-offs among accuracy, communication overhead, complexity, and the semantic proximity of AI-native communication devices.

## Simulations

This section provides the necessary commands to run the simulations required for the experiments. The commands execute different training scripts with specific configurations. Each simulation subsection contains both the `python` command and `uv` counterpart.

### Accuracy Vs Compression Factor

```bash
# Federated Semantic Alignment and Multi-Link Semantic Alignment
python scripts/train_linear.py communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 base_station.status=multi-link,shared simulation=compr_fact -m

# Baseline First-K
python scripts/train_baseline.py communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 base_station.strategy=First-K simulation=compr_fact -m

# Baseline Top-K
python scripts/train_baseline.py communication.channel_usage=1,2,3,4,5,10 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 base_station.strategy=Top-K simulation=compr_fact -m
```

```bash
# Federated Semantic Alignment and Multi-Link Semantic Alignment
uv run scripts/train_linear.py communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 simulation=compr_fact -m

# Baseline First-K
uv run scripts/train_baseline.py communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 base_station.strategy=First-K simulation=compr_fact -m

# Baseline Top-K
uv run scripts/train_baseline.py communication.channel_usage=1,2,3,4,5,10 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 base_station.strategy=Top-K simulation=compr_fact -m
```

### Accuracy Vs SNR

```bash
# Federated Semantic Alignment
python scripts/train_linear.py communication.channel_usage=1,4,8 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 simulation=snr -m

# Baseline First-K
python scripts/train_baseline.py communication.channel_usage=1,4,8 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 base_station.strategy=First-K simulation=snr -m

# Baseline Top-K
python scripts/train_baseline.py communication.channel_usage=2,4 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 base_station.strategy=Top-K simulation=snr -m
```

```bash
# Federated Semantic Alignment
uv run scripts/train_linear.py communication.channel_usage=1,4,8 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 simulation=snr -m

# Baseline First-K
x uv run scripts/train_baseline.py communication.channel_usage=1,4,8 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 base_station.strategy=First-K simulation=snr -m

# Baseline Top-K
uv run scripts/train_baseline.py communication.channel_usage=2,4 communication.antennas_receiver=4 communication.antennas_transmitter=4 seed=27,42,100,123,144,200 communication.snr=-20.0,-10.0,10.0,20.0,30.0 base_station.strategy=Top-K simulation=snr -m
```

### Heterogenous Vs Homogeneous

```bash
# Heterogeneous
python scripts/train_linear.py --config-name=heterogeneous communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 -m

# Homogeneous
python scripts/train_linear.py --config-name=homogeneous communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 -m
```

```bash
# Heterogeneous
uv run scripts/train_linear.py --config-name=heterogeneous communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 -m

# Homogeneous
uv run scripts/train_linear.py --config-name=homogeneous communication.channel_usage=1,2,4,6,8,10,20 communication.antennas_receiver=1,2,4 communication.antennas_transmitter=1,2,4 seed=27,42,100,123,144,200 -m
```

### Classifiers

The following command will initiate training of the required classifiers for the above simulations. However, this step is not strictly necessary, as the simulation scripts will automatically check for the presence of pretrained classifiers in the `models/classifiers` subfolder. If the classifiers are not found, a pretrained version (used in our paper) will be downloaded from Drive.

```bash
# Classifiers
python scripts/train_classifier.py rx_enc=vit_small_patch16_224,vit_small_patch32_224,vit_base_patch16_224,vit_base_patch32_clip_224,rexnet_100,mobilenetv3_small_075,mobilenetv3_large_100,mobilenetv3_small_100,efficientvit_m5.r224_in1k,levit_128s.fb_dist_in1k,vit_tiny_patch16_224 seed=27,42,100,123,144,200 -m
```

```bash
# Classifiers
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

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
@INPROCEEDINGS{dipoce2025fedsemalign,
  author={Di Poce, Giuseppe and Pandolfo, Mario Edoardo and Strinati, Emilio Calvanese and Di Lorenzo, Paolo},
  booktitle={2025 IEEE 26th International Workshop on Signal Processing and Artificial Intelligence for Wireless Communications (SPAWC)}, 
  title={Federated Latent Space Alignment for Multi-User Semantic Communications}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Wireless communication;Accuracy;Protocols;Equalizers;Noise;Key performance indicator;Semantic communication;Numerical models;Optimization;Semantic Communication;semantic equalization;latent space alignment;MIMO;Federated Learning},
  doi={10.1109/SPAWC66079.2025.11143294}
}
```

## Authors

- [Giuseppe Di Poce](https://github.com/giuseppedipoce)
- [Mario Edoardo Pandolfo](https://scholar.google.com/citations?user=wAeScL8AAAAJ&hl)
- [Emilio Calvanese Strinati](https://scholar.google.com/citations?user=bWndGhQAAAAJ)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?hl=en&user=VZYvspQAAAAJ)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
