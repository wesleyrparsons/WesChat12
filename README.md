# WesChat

**WesChat** is a homebrew transformer language model written primarily in **Free Pascal**, with **NVIDIA CUDA** and **cuBLAS** used for GPU computation.

The project is built largely from the ground up as a way to understand how modern language models actually work. Rather than relying on PyTorch, TensorFlow, or another machine-learning framework, WesChat implements the transformer training and inference pipeline directly.

This is an experimental and educational project, not a production LLM framework.

---

## Background

I started programming around 1970 using Fortran and BASIC in a time-sharing environment. My first compiler on a personal computer was **Turbo Pascal**, and Pascal has remained my language of choice ever since.

Most modern LLM development is done in Python using large machine-learning frameworks. WesChat deliberately takes a different approach. Tokenization, transformer operations, training, optimization, inference, model serialization, and the interface between Pascal and CUDA are implemented directly.

The goal is not to compete with modern LLM frameworks, but to understand what happens underneath them.

---

## Current Status

WesChat is **actively under development**.

The model now trains primarily on the GPU using CUDA and cuBLAS. Current development is focused on:

* Training stability
* AdamW optimization and diagnostics
* Adaptive learning-rate control
* CUDA performance
* Tokenizer comparisons
* Model checkpointing and reliable training resumption
* Training and inference diagnostics

The architecture and training code continue to evolve as experiments reveal better approaches.

---

## Model Architecture

WesChat implements a decoder-only autoregressive transformer.

The principal components are:

### Embeddings

* Learned token embeddings
* Transformer-style embedding scaling
* Weight tying between input embeddings and the output projection

### Multi-Head Self-Attention

* Multiple attention heads
* Causal/autoregressive masking
* Query, key, value, and output projections
* Rotary positional embeddings (RoPE)

### Layer Normalization

* Pre-LayerNorm transformer structure
* Learned gamma and beta parameters
* Cached normalization statistics for backward propagation
* Custom CUDA forward and backward kernels

### Feed-Forward Network

* Configurable expansion dimension, typically 4× the model dimension
* ReLU activation
* Learned input and output projections

### Regularization

Separate configurable dropout is available for:

* Attention
* MLP
* Residual paths

### Output Head

* Tied embedding/output weights
* Softmax probabilities
* Temperature control
* Cross-entropy loss
* Autoregressive token sampling

---

## Typical Model Configuration

WesChat is designed to allow model dimensions and training parameters to be changed between experiments. A representative current configuration is:

| Parameter           | Typical Value |
| ------------------- | ------------: |
| Model dimension     |           192 |
| Sequence length     |           256 |
| Attention heads     |             8 |
| Transformer blocks  |             6 |
| MLP expansion       |            4× |
| Precision           |       Float32 |
| Dropout             |          0.05 |
| Gradient clip limit |           0.4 |
| Weight decay        |        0.0001 |
| Optimizer           |         AdamW |
| Temperature         |           1.0 |

These are experimental settings rather than fixed architectural requirements.

---

## Tokenization

WesChat supports both its own tokenization system and GPT-2 tokenization.

### WesChat Tokenizer

The native tokenizer supports creation and use of WesChat symbol tables and tokenized corpora.

The tokenizer has evolved considerably during development and supports workflows such as:

* Creating symbol tables from corpora
* Tokenizing text using WesChat symbol tables
* Saving tokenized corpora
* Loading previously tokenized corpora for training
* Combining or working with externally generated tokenization data

### GPT-2 Tokenizer

WesChat can also use the standard GPT-2 vocabulary and merge table.

This provides a useful reference tokenizer and makes it possible to compare WesChat's native tokenization with an established byte-level BPE tokenizer.

---

## Training

Training uses a sliding window over the tokenized corpus.

The training loop performs:

1. Input and target token construction
2. Embedding lookup
3. Transformer forward pass
4. Output projection
5. Softmax and cross-entropy loss
6. Output-head backward pass
7. Transformer backward propagation
8. Gradient clipping
9. AdamW parameter updates

Window stride is configurable and windows can be shuffled between epochs.

Most large tensors remain on the GPU throughout training.

---

## AdamW Optimization

WesChat uses **AdamW** rather than simple gradient descent.

Optimizer state is maintained separately for trainable tensors, including the first and second moments required by Adam.

Training diagnostics include:

* Parameter RMS
* Update RMS
* Update/parameter ratio
* First-moment RMS
* Square-root second-moment RMS
* Gradient statistics
* LayerNorm gamma statistics

These diagnostics have proved useful for detecting unstable training long before a failure becomes obvious from loss alone.

---

## Learning Rate

WesChat supports several learning-rate modes, including:

* Fixed/flat learning rate
* Predefined learning-rate schedules
* Manual learning-rate override
* Adaptive learning-rate adjustment

The adaptive system uses current training and optimizer statistics to make conservative adjustments to the learning rate.

Learning-rate settings are retained in training checkpoints so that resumed training continues with the appropriate state.

---

## Loss and Training Statistics

WesChat reports mean cross-entropy loss over each training epoch.

Other diagnostics include:

* Mean epoch loss
* Change from the preceding epoch
* Perplexity
* Bits per byte
* Tokens per second
* AdamW statistics
* Parameter and gradient RMS statistics
* LayerNorm gamma statistics

---

## CUDA and cuBLAS

WesChat's computationally intensive operations are implemented using **NVIDIA CUDA** and **cuBLAS**.

cuBLAS is used for major matrix operations, while custom CUDA kernels implement operations specific to the transformer.

GPU operations include:

* Matrix multiplication
* Embedding lookup
* Softmax
* Cross-entropy loss
* Layer normalization
* LayerNorm backward propagation
* RoPE
* ReLU and ReLU backward
* Dropout and dropout backward
* Gradient clipping
* Bias operations
* Embedding-gradient accumulation
* AdamW parameter updates

The objective is to keep the training loop GPU-resident wherever practical and minimize unnecessary CPU↔GPU transfers.

---

## Gradient Clipping

Trainable parameter gradients can be clipped before the AdamW update.

This provides protection against occasional large gradients while allowing the normal AdamW update process to handle ordinary variation in gradient magnitude.

WesChat also reports clipping and gradient statistics as part of its training diagnostics.

---

## Model Files and Checkpointing

WesChat saves model weights together with information needed to resume training.

Current checkpoints can include:

* Model parameters
* AdamW first-moment state
* AdamW second-moment state
* Optimizer step
* Global training step
* Completed epoch count
* Learning-rate state
* Weight decay
* Gradient clipping settings
* Dropout settings
* Stride
* Random seed and other training settings

Older model formats can still be recognized, although older files may not contain sufficient optimizer state to resume AdamW training exactly where it stopped.

---

## Inference

WesChat can switch from training to autoregressive inference.

Inference supports:

* Prompt tokenization
* Autoregressive generation
* Temperature
* Top-K sampling
* Token probability diagnostics
* Repetition discouragement
* GPT-2 or WesChat token decoding, depending on the model/tokenizer configuration

Inference diagnostics can expose both raw model probabilities and probabilities after sampling adjustments.

---

## Corpus Experiments

WesChat has been trained on a variety of experimental corpora, including literary text and subsets of TinyStories.

These experiments are used to study:

* Training stability
* Learning-rate behavior
* Tokenization efficiency
* Model scaling
* CUDA performance
* Generalization and inference quality

The intent is experimental rather than benchmark-oriented.

---

## Project Organization

WesChat maintains separate working areas for items such as:

```text
corpus/
symbols/
tokens/
models/
logs/
runs/
scratch/
```

This separates source corpora, tokenizer data, token streams, trained models, and experimental output.

---

## Implementation Philosophy

WesChat intentionally avoids hiding the transformer behind a machine-learning framework.

Operations that would normally be a single PyTorch expression may instead involve:

* Explicit Pascal data structures
* CUDA device buffers
* cuBLAS calls
* Custom CUDA kernels
* Explicit forward and backward routines
* Explicit gradient storage
* Explicit optimizer state
* Explicit model serialization

That makes the program considerably more work to build, but that is also the point of the project.

The aim is to be able to trace a generated token all the way back through softmax, the output projection, transformer blocks, attention, normalization, embeddings, gradients, and parameter updates.

---

## Goals

The primary goals of WesChat are to:

* Understand transformer language models at a low level
* Implement an LLM without Python machine-learning frameworks
* Understand forward and backward propagation directly
* Explore CUDA and cuBLAS programming
* Experiment with tokenization
* Study optimization and training stability
* Maintain direct control over model memory and mathematics
* Build a complete working language model from first principles

---

## Development

WesChat remains experimental and is continuously changing.

Current areas of interest include:

* Improving training efficiency
* Refining adaptive learning-rate behavior
* Comparing WesChat and GPT-2 tokenization
* Improving checkpoint/resume behavior
* Monitoring optimizer and tensor health
* Improving inference quality
* Increasing model size as GPU memory and performance permit

---

## Closing

WesChat is a hands-on exploration of transformer language models implemented largely from first principles in Pascal and CUDA.

It is deliberately lower-level than conventional machine-learning projects. Rather than asking a framework to build and train a transformer, WesChat attempts to make each major operation visible and understandable.

For anyone interested in what happens beneath the abstractions of modern LLM frameworks—at the level of matrices, gradients, GPU memory, optimizer state, and individual CUDA kernels—that is the purpose of the project.
