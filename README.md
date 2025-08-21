# Transformer-Based Bigram Language Model (PyTorch)

This project is a **minimal PyTorch implementation of a Transformer-inspired Bigram Language Model**, based on the seminal paper *[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)*.

The implementation highlights how **self-attention mechanisms** (multi-head attention) can replace recurrence and convolution in sequence modeling tasks such as **language modeling**.

---

## 🚀 Features

* Implements a **character-level language model** trained on the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
* Components include:

  * Token & positional embeddings
  * Multi-Head Self-Attention (`Head`, `MultiHeadAttention`)
  * Position-wise Feed-Forward layers
* Trains with AdamW optimizer
* Text generation support (`generate` method)
* Runs on **GPU (CUDA)** if available

---

## 📂 Project Structure

```
├── Attention is All you Need.pdf   # Reference paper
├── v2.py                           # PyTorch implementation
├── input.txt                       # Training data (Tiny Shakespeare)
└── README.md                       # Documentation
```

---

## 🏃 Usage

### 1. Download training data

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

*(or replace with your own `input.txt` file)*

### 2. Train the model

```bash
python v2.py
```

### 3. Example training output

```
Step 0: Train Loss: 3.9021, Val Loss: 3.8765
Step 500: Train Loss: 2.3456, Val Loss: 2.3211
...
```

### 4. Generate text

The script automatically prints generated text at the end of training:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

---

## 📊 Model Overview

* **Embedding Dimension (`n_embd`)**: 32
* **Block Size (`block_size`)**: 8
* **Batch Size (`batch_size`)**: 32
* **Learning Rate**: 1e-3
* **Optimizer**: AdamW
* **Attention Heads**: 4 heads × 8-dim each

---

## 📖 Reference

* Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 【Attention is All you Need.pdf】
