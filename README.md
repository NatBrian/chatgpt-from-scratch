# Building a Modern Large Language Model from Scratch

This repository contains a comprehensive, self-contained Jupyter notebook designed to teach the fundamental architecture and training mechanics of modern Large Language Models (LLMs).

It provides a step-by-step implementation of a Generative Pre-trained Transformer (GPT), transitioning from basic language modeling concepts to a fully functional model incorporating state-of-the-art architectural components found in systems like Llama 3.

## Project Overview

The primary objective of this project is educational: to demystify the internal operations of LLMs by building one from first principles. Rather than relying on high-level abstractions from libraries like `transformers`, this notebook implements every component using only Python and PyTorch.

### Key Learning Objectives
*   **Tokenization:** Implementation of Byte Pair Encoding (BPE) for efficient text representation.
*   **Transformer Architecture:** A detailed construction of the Transformer block, including:
    *   **Causal Self-Attention:** From naive loops to efficient matrix operations.
    *   **Multi-Head Attention:** Parallelizing attention mechanisms.
*   **Modern Architectural Enhancements:** Integrating components standard in 2024/2025 LLM architectures:
    *   **RMSNorm** (Root Mean Square Normalization) for improved training stability.
    *   **RoPE** (Rotary Positional Embeddings) for better relative position handling.
    *   **SwiGLU** (Swish-Gated Linear Unit) for enhanced expressivity in feed-forward layers.
*   **Training Infrastructure:** Developing a robust training loop with mixed-precision training (float16), gradient accumulation, and checkpointing.
*   **Inference & Alignment:** Implementing generation strategies (temperature, top-k sampling) and introducing Reinforcement Learning from Human Feedback (RLHF) concepts.

## Target Audience

This resource is intended for software engineers, data scientists, and researchers seeking a deep, code-level understanding of LLMs.

**Prerequisites:**
*   Proficiency in Python programming.
*   Foundational knowledge of PyTorch (tensor operations, basic neural network modules).
*   A Google account (for execution on Google Colab).

## Notebook Structure

The `nanochat_zero_to_hero.ipynb` notebook follows a progressive structure:

1.  **Baseline Language Model:** Implementing a simple Bigram model to establish a performance baseline.
2.  **Data Processing:** Building the BPE tokenizer pipeline.
3.  **Attention Mechanisms:** Deriving self-attention mathematically and implementing it efficiently.
4.  **Architectural Upgrades:** Replacing standard Transformer components with Llama-style improvements (RMSNorm, RoPE, SwiGLU).
5.  **Model Assembly:** Composing the full GPT architecture (~30M parameters).
6.  **Model Training:** Training on the `FineWeb-Edu` dataset using a T4 GPU.
7.  **Inference:** Implementing the generation loop and chat interface.
8.  **Alignment (RLHF):** A conceptual introduction to aligning model outputs with human intent.

## Usage Instructions

This project is optimized for the Google Colab Free Tier.

### 1. Environment Setup
Open the notebook in Google Colab by clicking the badge above or uploading `nanochat_zero_to_hero.ipynb` to your Google Drive.

### 2. Hardware Acceleration
Ensure a GPU runtime is selected to enable efficient training:
*   Navigate to **Runtime** > **Change runtime type**.
*   Select **T4 GPU**.

### 3. Execution
Execute the notebook cells sequentially. The notebook is designed to be idempotent and self-contained.
*   **Runtime:** Approximately 2 hours for a complete end-to-end run.
*   **Persistence:** The notebook implements automatic checkpointing to Google Drive, allowing training to resume seamlessly in case of session interruption.

## Scope and Limitations

*   **Educational Focus:** The resulting model (~30M parameters) is designed for pedagogical clarity and rapid iteration, not for production-grade performance or reasoning capabilities comparable to commercial LLMs.
*   **Implementation Depth:** While the architecture is modern, the scale is deliberately constrained to fit within free cloud resources.

## Attribution and References

This project is inspired by the educational materials of **Andrej Karpathy**, specifically:
*   ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
*   [nanoGPT Repository](https://github.com/karpathy/nanoGPT)

**Distinction from nanoGPT:**
While `nanoGPT` serves as a lightweight training repository for standard GPT-2 architectures, this project is a pedagogical notebook that updates the architecture to reflect modern standards (Llama 3), specifically by substituting LayerNorm, ReLU, and absolute embeddings with RMSNorm, SwiGLU, and RoPE.

## Future Directions

Upon completion, learners are encouraged to:
1.  **Scale:** Increase model size (e.g., to 124M parameters) by adjusting the `COLAB_MODE` flag.
2.  **Data:** Retrain the model on domain-specific datasets.
3.  **Deep Dive:** Explore production codebases such as `llama.cpp` or Hugging Face `transformers` with a grounded understanding of their internal mechanics.