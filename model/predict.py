#!/usr/bin/env python3
import struct
import string
import sys
import os
import numpy as np

MAX_WORD_LEN = 64
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def tokenize(text):
    return text.lower().translate(PUNCT_TABLE).split()

def load_vocab(path=None):
    if path is None:
        path = os.path.join(SCRIPT_DIR, "vocab.bin")
    word2idx = {}
    with open(path, "rb") as f:
        vocab_size = struct.unpack("I", f.read(4))[0]
        for i in range(vocab_size):
            raw = f.read(MAX_WORD_LEN)
            word = raw.split(b"\x00", 1)[0].decode("utf-8")
            if word:
                word2idx[word] = i
    return word2idx


def load_weights(path=None):
    if path is None:
        path = os.path.join(SCRIPT_DIR, "weights.bin")
    with open(path, "rb") as f:
        vocab_size, embed_dim, hidden_dim, output_dim = struct.unpack("IIII", f.read(16))

        embedding  = np.frombuffer(f.read(vocab_size * embed_dim * 4),  dtype=np.float32).reshape(vocab_size, embed_dim)
        fc1_weight = np.frombuffer(f.read(embed_dim * hidden_dim * 4),  dtype=np.float32).reshape(embed_dim, hidden_dim)
        fc1_bias   = np.frombuffer(f.read(hidden_dim * 4),              dtype=np.float32)
        fc2_weight = np.frombuffer(f.read(hidden_dim * output_dim * 4), dtype=np.float32).reshape(hidden_dim, output_dim)
        fc2_bias   = np.frombuffer(f.read(output_dim * 4),              dtype=np.float32)

    return {
        "embedding": embedding,
        "fc1_weight": fc1_weight, "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight, "fc2_bias": fc2_bias,
    }

def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def predict(text, weights, word2idx):
    indices = [word2idx.get(w, 0) for w in tokenize(text)]
    if not indices:
        indices = [0]
    mean_embed = weights["embedding"][indices].mean(axis=0)

    hidden = np.maximum(0, mean_embed @ weights["fc1_weight"] + weights["fc1_bias"])

    output = sigmoid(hidden @ weights["fc2_weight"] + weights["fc2_bias"])

    return float(output[0])


# ── Mainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn 

def main():
    word2idx = load_vocab()
    weights = load_weights()

    args = [a for a in sys.argv[1:] if a != "-f"]

    if args and "-f" not in sys.argv:
        text = " ".join(args)
        conf = predict(text, weights, word2idx)
        label = "Positive" if conf > 0.5 else "Negative"
        print(f"{label} ({conf:.4f})")
        return

    if "-f" in sys.argv:
        filepath = sys.argv[sys.argv.index("-f") + 1]
        with open(filepath) as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                conf = predict(text, weights, word2idx)
                label = "Positive" if conf > 0.5 else "Negative"
                print(f"{label} ({conf:.4f}) | {text}")
        return

    print("Model loaded. Type a sentence, or 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        conf = predict(text, weights, word2idx)
        label = "Positive" if conf > 0.5 else "Negative"
        bar = "+" * int(conf * 20) + "-" * (20 - int(conf * 20))
        print(f"  {label} ({conf:.4f})  [{bar}]")


if __name__ == "__main__":
    main()
