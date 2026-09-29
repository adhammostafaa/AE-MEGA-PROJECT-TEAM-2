# AE-MEGA-PROJECT-TEAM-2

## AI Team:
### Adham Rahhal:
* did the python program to generate the odlc dataset containing 1200 different images
* did the transformer model implementation: -multi head self attention from scratch -causal attention mask -feed forward neural network -positional encodings -model training and training loop -logging the training and validation losses and validation perplexity
# Transformer-Based Text Generation Pipeline

A from-scratch, GPT-style causal Transformer language model — implemented in pure PyTorch (`nn.Linear`, `nn.Embedding`, no pre-built Transformer modules) — trained on Simple English Wikipedia and used for autoregressive text generation with temperature/top-k/top-p sampling and attention visualization.

## Overview

| Stage | What it does |
|---|---|
| 2.1 Data Gathering & Preprocessing | Loads and cleans a Wikipedia text corpus, tokenizes it, and chunks it into fixed-length blocks |
| 2.2 Transformer Model | Implements a causal decoder Transformer from scratch and trains it with AdamW |
| 3. Inference & Generation | Custom autoregressive generation function with temperature, top-k/top-p sampling, and repetition penalty |
| Bonus | Attention heatmap visualization of a trained head |

## 1. Data Gathering & Preprocessing

- **Source:** [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.simple` config — Simple English Wikipedia), `train` split.
- **Sampling:** shuffled (`seed=42`) and reduced to 120,000 articles, then split 90/5/5 into train / validation / test. `id`, `url`, and `title` columns are dropped.
- **Cleaning (`clean`):** strips trailing sections such as *References*, *Other websites*, *Related pages*, *Sources*, *Further reading*, *External links* via regex; truncates each article to 20,000 characters; collapses whitespace.
- **Filtering (`is_valid`):** drops empty articles and anything shorter than 300 characters after cleaning.
- **Tokenizer:** pretrained GPT-2 byte-pair encoding (`AutoTokenizer.from_pretrained("gpt2")`), with `pad_token` set to `eos_token`. Each article's token sequence has an explicit EOS token appended so the model can learn article boundaries.
- **Blocking:** all tokens are concatenated and re-chunked into fixed-length blocks of `block_size = 128` tokens for next-token-prediction training; `labels` are a copy of `input_ids`.

**Resulting dataset sizes (128-token blocks):**

| Split | Rows |
|---|---|
| train | 166,277 |
| validation | 9,439 |
| test | 9,219 |

## 2. Transformer Model

A GPT-style causal decoder built entirely from base PyTorch layers:

- **`MultiHeadAttention`** — separate `Q`/`K`/`V` linear projections, scaled dot-product attention with a causal mask (`torch.tril`). Uses PyTorch's fused `scaled_dot_product_attention` by default (`use_sdpa = True`); a manual (`Q·Kᵀ`/softmax) path is also implemented and used for the attention-weight visualization.
- **`TransformerBlock`** — pre-LayerNorm, residual multi-head attention + a GELU feed-forward network (4× expansion).
- **`CausalTransformer`** — learned token + positional embeddings, a stack of `TransformerBlock`s, final LayerNorm, and an output projection whose weights are tied to the token embedding.

**Hyperparameters used:**

| Param | Value |
|---|---|
| `d_model` | 384 |
| `num_heads` | 6 |
| `num_layers` | 8 |
| `block_size` (context length) | 128 |
| `dropout` | 0.1 |
| `batch_size` | 32 |
| `num_epochs` | 5 |
| `learning_rate` | 5e-4 (AdamW, cosine annealing schedule) |

**Training setup:** cross-entropy loss (next-token prediction), AdamW (`betas=(0.9, 0.95)`, `weight_decay=0.1`), gradient clipping (`max_norm=1.0`), mixed-precision (`torch.autocast` + `GradScaler`) on GPU, cosine-annealing LR schedule, early stopping on validation loss (`patience=3`). The best checkpoint (by validation loss) is saved automatically each epoch it improves.

**Training results:**

| Epoch | Train Loss | Val Loss | Val Perplexity |
|---|---|---|---|
| 1 | 5.4276 | 4.7798 | 119.08 |
| 2 | 4.6207 | 4.4360 | 84.44 |
| 3 | 4.3672 | 4.2818 | 72.37 |
| 4 | 4.2075 | 4.1726 | 64.88 |
| 5 | 4.0908 | 4.1013 | **60.42 (best)** |

**Test set:** loss `4.1073`, perplexity `60.78`.

Loss and perplexity curves are plotted and saved to `transformer_training_metrics.png`.

## 3. Inference & Token Generation

`generate_text(prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty)`:

1. Tokenizes the input prompt with the GPT-2 tokenizer (no special tokens added).
2. Autoregressively predicts one token at a time, feeding the growing sequence back into the model (context is cropped to the last `block_size` tokens).
3. Applies **temperature scaling** to the logits, then optional **top-k** filtering, optional **top-p (nucleus)** filtering, and a **repetition penalty** that down-weights tokens already seen in context.
4. Samples the next token from the resulting distribution (`torch.multinomial`).
5. Detokenizes the full sequence back to text once generation finishes, replacing the EOS token with a newline.

Example output (`temperature=0.8`, prompt `"The history of science"`):

> *The history of science is not a collection of information. It was discovered in 1843 by James R. Johnson. It is now in the National Museum of Science and Sciences*

## Bonus: Attention Heatmap Visualization

`get_attention_weights(token_ids, layer, head)` hooks a chosen `TransformerBlock`'s attention module, recomputes that head's `Q·Kᵀ` scores with the causal mask applied, and returns the resulting attention-weight matrix. This is plotted as a token-by-token heatmap to show which prior tokens the model attends to when predicting each next token — useful for inspecting patterns like local/recency attention or attention "sinks" on the first token.

## Saving & Loading

- **Save:** model weights + config (`vocab_size`, `d_model`, `num_heads`, `num_layers`, `max_seq_len`, `dropout`) are saved to `saved_model/causal_transformer.pt`; the tokenizer is saved to `saved_model/tokenizer/`.
- **Load:** reconstruct `CausalTransformer(**checkpoint["model_config"])`, load the state dict, and reload the tokenizer from the saved directory — avoids retraining (training all 5 epochs takes a non-trivial amount of GPU time).

## Requirements

`torch`, `transformers`, `datasets`, `matplotlib` (all pre-installed on Colab; no extra `pip install` needed for this part).
