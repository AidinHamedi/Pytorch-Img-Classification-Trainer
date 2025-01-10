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

There is an example already made that uses this system to run experiments (hyper parameters tuning), The experiments params are set in the `./expers.toml` file and the experiment runner is `./run_expers.py`. </br>
The experiment runner will run the `train` function in `./train_exper.py` with the experiment params as the arg in that function you can set what the params do and etc... </br>
In end you will see that the fit function of the `Training_Engine` is being called and here the magic happens 😊.

## 📚 Documentation

### Training Engine

You can access the main `fit` function from `./Training_Engine/trainer.py` file.
The `fit` function takes the following **required** arguments:

- `model`: The model to be trained.
- `train_dataloader`: The training data loader. (**DynamicArg**)
- `test_dataloader`: The test data loader. (**DynamicArg**)
- `optimizer`: The optimizer to be used for training.
- `loss_fn`: The loss function to be used for training.

And done all of the other args are optional and used setting mixed precision and etc... </br>
I think you have noticed that the `train_dataloader` and `test_dataloader` are **DynamicArg** so what is a DynamicArg?

### DynamicArg

A DynamicArg is a special type of argument that allows you to pass a function as an generator and outputs a value based on the the environment. </br>
Like for example you make the `train_dataloader` DynamicArg return a pytorch dataloader that adjusts the augmentation amount based on the epoch count.
Its not that complicated just by looking at the code you will understand it. </br>
You can import dynamic args from `./Training_Engine/Utils/Base/dynamic_args.py`.

### Utils

There are some utils with the training engine that you can use for like loading the images and etc...

- `./Training_Engine/Utils/Data/data_loader.py`: A utility for loading images from a directory or making a pytorch dataset for loading large datasets on the fly.
- `./Training_Engine/Utils/Data/normalization.py`: A utility for normalizing images and getting class weighting.

## 📷 Example Output

![Img](doc\example.png)

## 📝 License

<pre>
 Copyright (c) 2025 Aidin Hamedi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
</pre>
