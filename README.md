# WesChat

**WesChat** is a homebrew large language model written in **Free Pascal**, built from the ground up to understand how modern LLMs actually work.

This is not a production system—it’s an educational, experimental project aimed at going *under the hood* of transformer-based models rather than relying on high-level frameworks.

---

## Background

I started programming around 1970 using Fortran and BASIC in a time-sharing environment. My first compiler on a personal computer was **Turbo Pascal**, and Pascal has remained my language of choice ever since.

Most modern LLM work is done in Python using large frameworks. WesChat takes a different path: implementing the entire pipeline—from tokenization to training—in Pascal, with direct control over memory, data structures, and numerical operations.

---

## Status

WesChat is **actively under development**. There is no release candidate yet.

Current focus:

- Migrating computation from CPU to **NVIDIA CUDA**
- Using **cuBLAS** for matrix operations
- Writing custom CUDA kernels for core transformer steps

A flow chart of the program is available in flowchart.pas.

---

## Features

### Tokenization

WesChat includes a custom **byte-level BPE tokenizer** with:

- Deterministic left-to-right longest-prefix matching
- Greedy longest-match decoding
- Compatibility with external vocab/merge formats (GPT-style)

Supported workflows:

- Tokenize a single corpus
- Tokenize multiple corpora (file list → concatenated tokens)
- Tokenize using:
  - WesChat symbol tables
  - External symbol + merge tables
- Combine symbol tables
- Generate symbol tables from a corpus

---

## Training Pipeline

After tokenization, WesChat trains a transformer model:

- Sliding window over token stream
- Configurable stride
- Forward + backward passes
- Gradient-based updates

---

## Model Architecture

WesChat implements a compact transformer:

- 4–8 transformer blocks
- 8 attention heads
- Model dimension: 256
- Sequence length: 128 or 256
- Float32 precision

### Components

**Attention**
- Multi-head self-attention
- Autoregressive masking
- Rotary positional encoding (RoPE)

**Normalization**
- Pre-layer normalization
- Cached statistics for backprop

**MLP**
- 4× expansion
- ReLU activation

**Regularization**
- Dropout (attention, residual, MLP): 0.1

**Output**
- Softmax with temperature = 1.0
- Cross-entropy loss
- Weight tying (embeddings ↔ output)

---

## Hyperparameters

| Parameter       | Value |
|----------------|------|
| ModelDim       | 256  |
| SeqLen         | 128 / 256 |
| Heads          | 8 |
| Dropout        | 0.1 |
| Learning Rate  | 0.01 |
| Temperature    | 1.0 |

---

## CUDA Acceleration (Work in Progress)

WesChat is being migrated to GPU execution:

- cuBLAS for GEMM operations
- Custom CUDA kernels for:
  - LayerNorm
  - Softmax
  - Dropout
  - RoPE
  - Embedding lookup

Goal: fully GPU-resident training loop with minimal CPU↔GPU transfer.

---

## File Outputs

All outputs are written to a time-stamped directory, which includes

- Model weights
- Symbol tables
- Tokenized corpus (optional)

---

## Goals

- Understand transformer internals at a low level
- Build an LLM without Python frameworks
- Explore CPU vs GPU performance tradeoffs
- Maintain full control over memory and math

---

## Notes

This project is experimental and evolving. Active areas of work:

- CUDA integration
- Training stability
- Model serialization

---

## Closing

WesChat is a hands-on exploration of how large language models actually work—implemented in a language that predates most modern ML tooling.

If you’re interested in transformers at the level of matrices, memory, and math, this project may be useful—or at least interesting.
