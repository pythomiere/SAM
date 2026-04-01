import string
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from export import export_vocab, export_weights

# ── Hyperparameters

VOCAB_SIZE = 8000
EMBED_DIM  = 64
HIDDEN_DIM = 32
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 0.001

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(text):
    return text.lower().translate(PUNCT_TABLE).split()

def build_vocab(sentences):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))

    word2idx = {"<UNK>": 0}
    for i, (word, _) in enumerate(counter.most_common(VOCAB_SIZE), start=1):
        word2idx[word] = i

    return word2idx


def encode(text, word2idx):
    return [word2idx.get(w, 0) for w in tokenize(text)]

class SentimentDataset(Dataset):
    def __init__(self, sentences, labels, word2idx):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        indices = encode(self.sentences[idx], self.word2idx)
        if not indices:
            indices = [0]  # fallback to <UNK>
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    offsets = [0]
    combined = []
    for t in texts:
        combined.append(t)
        offsets.append(offsets[-1] + len(t))
    offsets = offsets[:-1]
    return (
        torch.cat(combined),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float),
    )

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)   # mean of word embeddings
        h = torch.relu(self.fc1(embedded))         # hidden layer
        return torch.sigmoid(self.fc2(h)).squeeze(1)

def train():
    dataset = load_dataset("stanfordnlp/sst2")

    train_sentences = dataset["train"]["sentence"]
    train_labels    = dataset["train"]["label"]
    val_sentences   = dataset["validation"]["sentence"]
    val_labels      = dataset["validation"]["label"]
    word2idx = build_vocab(train_sentences)
    vocab_size = len(word2idx)
    train_loader = DataLoader(
        SentimentDataset(train_sentences, train_labels, word2idx),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SentimentDataset(val_sentences, val_labels, word2idx),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    )

    model = SentimentModel(vocab_size, EMBED_DIM, HIDDEN_DIM)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for text, offsets, labels in train_loader:
            optimizer.zero_grad()
            preds = model(text, offsets)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += ((preds > 0.5).float() == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for text, offsets, labels in val_loader:
                preds = model(text, offsets)
                val_correct += ((preds > 0.5).float() == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    export_vocab(word2idx, "vocab.bin")
    export_weights(model, "weights.bin")

if __name__ == "__main__":
    train()
