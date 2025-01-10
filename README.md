# Pytorch Image Classification Trainer

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A Pytorch image classification training system, easy to use (relatively 😅)** </br>
Supports **Tensorboard** logging and techniques like **Gradient centralization**, **Adaptive gradient clipping** and etc...

## 🚀 Getting started

### Step 1: Clone the repository

```bash
git clone https://github.com/AidinHamedi/Pytorch-Img-Classification-Trainer.git
```

### Step 2: Install the requirements

This repo uses [**uv**](https://github.com/astral-sh/uv) to manage it dependencies.

```bash
uv sync
```

### Step 3: Check out the example

There is an example are made that uses this system to run experiments (hyper parameters tuning), The experiments params are set in the `./expers.toml` file and the experiment runner is `./run_expers.py`. </br>
The experiment runner will run the `train` function in `./train_exper.py` with the experiment params as the arg in that function you can set what the params do and etc... </br>
In end you will see that the fit function of the `Training_Engine` is being called heres when the magic happens.


## 📝 License

<pre>
 Copyright (c) 2025 Aidin Hamedi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
</pre>
