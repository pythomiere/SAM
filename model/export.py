"""
DISCLAIMER: this file is written by Claude
"""
import struct
import numpy as np

MAX_WORD_LEN = 64


def export_vocab(word2idx, path="vocab.bin"):
    vocab_size = len(word2idx)

    idx2word = [""] * vocab_size
    for word, idx in word2idx.items():
        idx2word[idx] = word

    with open(path, "wb") as f:
        f.write(struct.pack("I", vocab_size))
        for word in idx2word:
            raw = word.encode("utf-8")[:MAX_WORD_LEN - 1]
            f.write(raw.ljust(MAX_WORD_LEN, b"\x00"))

    print(f"  vocab.bin  -> {vocab_size} words")


def export_weights(model, path="weights.bin"):
    vocab_size, embed_dim = model.embedding.weight.shape
    hidden_dim = model.fc1.out_features
    output_dim = model.fc2.out_features

    with open(path, "wb") as f:
        f.write(struct.pack("IIII", vocab_size, embed_dim, hidden_dim, output_dim))
        f.write(model.embedding.weight.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc1.weight.detach().cpu().numpy().T.astype(np.float32).tobytes())
        f.write(model.fc1.bias.detach().cpu().numpy().astype(np.float32).tobytes())
        f.write(model.fc2.weight.detach().cpu().numpy().T.astype(np.float32).tobytes())
        f.write(model.fc2.bias.detach().cpu().numpy().astype(np.float32).tobytes())

    total_params = (vocab_size * embed_dim
                    + embed_dim * hidden_dim + hidden_dim
                    + hidden_dim * output_dim + output_dim)
    size_mb = (16 + total_params * 4) / 1024 / 1024
    print(f"  weights.bin -> {total_params} params ({size_mb:.2f} MB)")
