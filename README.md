# Learning Large Language Models: A From-Scratch Transformer Implementation

This repository contains a Jupyter Notebook that provides a complete, step-by-step implementation of a Transformer-based GPT model from scratch using PyTorch. It's designed for educational purposes to demystify the architecture behind modern Large Language Models.

## Table of Contents

- [Project Overview](#project-overview)
- [The Transformer Architecture](#the-transformer-architecture)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Project Overview

The goal of this project is to provide a clear and concise implementation of the Transformer architecture, as introduced in the paper "Attention Is All You Need" and popularized by models like GPT. The notebook `Complete_Implementation_of_Transformer_Architecture.ipynb` walks through the entire process:

1. **Data Loading & Tokenization**: Preparing a text corpus and converting it into tokens suitable for the model.
2. **Model Components**: Building the fundamental blocks of the Transformer, including Multi-Head Attention, Layer Normalization, and Feed-Forward networks.
3. **GPT Model Assembly**: Combining the building blocks into a full GPT-style decoder model.
4. **Training**: Implementing a training loop to train the model on a dataset.
5. **Text Generation**: Using the trained model to generate new text, exploring various decoding strategies like Greedy Search, Temperature Scaling, and Top-K Sampling.

## The Transformer Architecture

This implementation focuses on a decoder-only Transformer architecture similar to GPT. The key components you will find implemented from scratch in the notebook are:

- **Token and Positional Embeddings**: To represent input tokens and their positions in the sequence.
- **Multi-Head Self-Attention**: The core mechanism that allows the model to weigh the importance of different tokens in the input sequence.
- **Layer Normalization and Skip Connections**: Essential for stabilizing the training of deep networks.
- **Position-wise Feed-Forward Networks**: Applied to each position separately and identically.
- **The Transformer Block**: An encapsulation of the attention and feed-forward mechanisms.
- **Final GPT Model**: The complete model that stacks multiple Transformer blocks.

## Key Features

- **From-Scratch Implementation**: Core components like `MultiHeadAttention`, `LayerNorm`, and `GELU` are built from basic PyTorch operations for maximum clarity.
- **Educational Focus**: The code is extensively commented, and the notebook is structured to be followed sequentially, making it an excellent learning resource.
- **Complete Pipeline**: Covers the entire process from data preparation and tokenization to training and inference.
- **Decoding Strategies**: Includes clear implementations and explanations of Greedy Decoding, Temperature Scaling, and Top-K Sampling to control the creativity and coherence of the generated text.

## Dependencies

To run the notebook, you need to have Python 3 and the following libraries installed:

- `torch`
- `tiktoken`
- `matplotlib`

You can install them using pip:

```bash
pip install torch tiktoken matplotlib
```

## Usage

1. **Clone the repository:**

```bash
git clone <repository-url>
cd Learn_Large_Languge_Models
```

2. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Jupyter Notebook:**

```bash
jupyter notebook Complete_Implementation_of_Transformer_Architecture.ipynb
```

From there, you can run the cells sequentially to see the model being built, trained, and used for text generation.

## Acknowledgments

- The Transformer architecture was introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.
- The code and structure are heavily inspired by the work of [Sebastian Raschka](https://github.com/rasbt) in his "LLMs-from-scratch" project.

---

_**Contact**_

[Facebook](https://www.facebook.com/ShifatHasanGNS/)
&emsp;
[Instagram](https://www.instagram.com/ShifatHasanGNS/)
&emsp;
[Linkedin](https://www.linkedin.com/in/md-shifat-hasan-8179402b4/)
&emsp;
[X](https://x.com/ShifatHasanGNS)
