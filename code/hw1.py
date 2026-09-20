"""Complete the TODOs. Preserve the supplied interfaces."""
from __future__ import annotations

from collections import Counter
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

TAGS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
PAD, UNK = 0, 1


def get_data(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL, ignoring blank lines, preserving sentence order.

    Each record has unique nonempty string `id`, nonempty list[str] `tokens`,
    and equal-length list[str] `tags` drawn from TAGS. Tokens must be nonempty
    and contain no whitespace. Raise ValueError for invalid records, duplicate
    IDs, invalid JSON, or empty files; let FileNotFoundError propagate.
    Return only the three required fields; additional fields may be ignored.
    Example record: {"id":"s1","tokens":["Birds","fly"],"tags":["NOUN","VERB"]}.
    """

    def validate_record(record):
        ALLOWED_TAGS = {'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', "PART", 'PRON', 'PROPN',
                        "PUNCT", 'SCONJ', 'SYM', 'VERB', 'X'}
        # Unique nonempty ID
        if type(record) != dict:
            raise ValueError("Bad Record")
        if not record.get('id', None):
            raise ValueError("No ID")
        if type(record.get('id')) != str:
            raise ValueError("Wrong ID type")
        if type(record.get('tokens')) != list or (
                len(record.get('tokens')) > 0 and type(record.get('tokens')[0]) != str):
            raise ValueError("Wrong token type")
        # Nonempty tokens list
        if not record.get('tokens') or len(record.get('tokens')) < 0:
            raise ValueError("Tokens are malformed")
        # Equal length tags
        if not record.get('tags') or len(record.get('tags')) != len(record.get('tokens')):
            raise ValueError("Tags are malformed")
        # Known tags
        for tag in record.get('tags'):
            if tag not in ALLOWED_TAGS:
                raise ValueError("Unknown tags")
        # Tokens don't have whitespace
        for token in record.get('tokens'):
            if not len(token):
                raise ValueError('Token is empty')
            if len(token) != len(token.strip()):
                raise ValueError('Token contains whitespace')
            for t in token:
                if t.isspace():
                    raise ValueError('Token contains whitespace')

    lines = []
    with open(path) as f:
        for raw_line in f:
            if raw_line and raw_line.strip():
                try:
                    line = json.loads(raw_line.strip())
                except JSONDecodeError:
                    raise ValueError("JSON decode error")
            else:
                continue
            validate_record(line)
            # Only keep these 3 fields
            filtered_line = {"id": line.get('id'), "tokens": line.get('tokens'), "tags": line.get('tags')}
            lines.append(filtered_line)
        if not len(lines):
            raise ValueError("File is empty")

        # Check all IDs are unique
        ids = [line['id'] for line in lines]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs")

        return lines


def build_vocab(records: list[dict], min_freq: int = 2) -> dict[str, int]:
    """Build a lowercase training vocabulary: PAD=0, UNK=1, then frequent words in sorted order."""
    # Get words list
    words = [word.lower() for record in records for word in record['tokens']]
    counts = {}
    # Build counts
    for word in words:
        if counts.get(word):
            counts[word] += 1
        else:
            counts[word] = 1
    # Skip < min_freq
    for word, freq in list(counts.items()):
        if freq < min_freq:
            counts.pop(word)
    # Sort by frequency
    # sorted_words = sorted(counts, key=counts.get, reverse=True)
    sorted_words = sorted(counts)
    # Build vocab dict starting at 2
    vocab = {str(word): id + 2 for id, word in enumerate(sorted_words)}
    # Add extra tokens
    vocab["<PAD>"], vocab["<UNK>"] = 0, 1
    return vocab


def encode_words(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """Map lowercase words to IDs, using UNK for unseen words."""
    token_ids = []
    for token in tokens:
        token_ids.append(vocab.get(token.lower(), UNK))
    return token_ids


def make_windows(values: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Zero-pad sentence windows: [N] or [N,D] -> [N,2*radius+1] or [N,2*radius+1,D]."""
    total_r = 2 * radius + 1
    if values.ndim == 1:
        full_row = ([0] * radius)
        full_row.extend(values)
        full_row.extend([0] * radius)
        padded = []
        for i in range(len(values)):
            # Shift sliding window each time
            row = full_row[i:2 * radius + 1 + i]
            padded.append(torch.tensor(row, dtype=values.dtype))
        return torch.stack(padded)
    else:
        # 2D vector
        count = values.shape[0]  # 7
        height = 2 * radius + 1  # 5
        width = values.shape[1]  # 3
        full_tensor = torch.zeros(count + 2 * radius, width, dtype=values.dtype)
        full_tensor[radius:radius + count] = values
        res = torch.zeros(count, height, width, dtype=values.dtype)
        for idx in range(count):
            res[idx] = full_tensor[idx:2 * radius + 1 + idx]
        return res


class LinguisticFeatures:
    """Three suffix one-hot blocks followed by five binary word features."""

    def __init__(self):
        self.suffix_vocabs: list[dict[str, int]] = []

    def fit(self, records: list[dict]) -> "LinguisticFeatures":
        """Fit suffix vocabularies of lengths 1, 2, and 3 on training data; return self."""
        for i in range(3):
            self.suffix_vocabs.append({})
        for idx, vocab in enumerate(self.suffix_vocabs):
            suffix_len = idx + 1
            # Build counts dict
            counts = {}
            for record in records:
                for word in record['tokens']:
                    word = word.lower()
                    if len(word) < suffix_len:
                        continue
                    suffix = word[-suffix_len:]
                    if suffix in counts:
                        counts[suffix] += 1
                    else:
                        counts[suffix] = 1
            filtered_counts = {}
            for word, count in counts.items():
                if count >= 2:
                    filtered_counts[word] = count
            sorted_words = sorted(filtered_counts.keys())
            for i, word in enumerate(sorted_words):
                vocab[word] = i + 1
            vocab['<UNK>'] = 0
        return self

    @property
    def dim(self) -> int:
        return sum(len(vocab) for vocab in self.suffix_vocabs) + 5

    def transform(self, tokens: list[str]) -> torch.Tensor:
        """Return float32 features [N,dim] without changing fitted vocabularies."""
        vocab_width = sum(len(vocab) for vocab in self.suffix_vocabs)
        total_width = vocab_width + 5
        res = []
        for i, token in enumerate(tokens):
            token_tens = [0] * total_width
            idx = 0
            for suffix_idx, vocab in enumerate(self.suffix_vocabs):
                suffix_len = suffix_idx + 1

                suffix = token[-suffix_len:].lower()
                if suffix in vocab:
                    token_tens[idx + vocab[suffix]] += 1
                else:
                    token_tens[idx] += 1  # <UNK>
                idx += len(vocab)
            flag_start_idx = vocab_width
            # First char uppercase
            if token[0].isupper():
                token_tens[flag_start_idx] = 1
            # Whole word uppercase
            if token.isupper():
                token_tens[flag_start_idx + 1] = 1
            # Word contains digit, Word contains hyphen
            for c in token:
                if c.isdigit():
                    token_tens[flag_start_idx + 2] = 1
                if c == '-':
                    token_tens[flag_start_idx + 3] = 1

            # Word is first in sentence
            if i == 0:
                token_tens[flag_start_idx + 4] = 1

            res.append(token_tens)
        return torch.Tensor(res).reshape(len(res), total_width)


class POSMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_tags: int = 17, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class WindowTagger(nn.Module):
    """Word-window classifier for models 1, 2, and 3."""

    def __init__(self, vocab_size: int, radius: int = 2, feature_dim: int = 0,
                 embedding_dim: int = 100, hidden_dim: int = 128,
                 num_tags: int = 17, dropout: float = 0.1):
        super().__init__()
        self.radius = radius
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # nn.Embedding
        self.count = 2 * radius + 1
        self.input_dim = self.count * embedding_dim + feature_dim  # num words * word size (embedding) + flags
        self.classifier = POSMLP(input_dim=self.input_dim, hidden_dim=hidden_dim, num_tags=num_tags)
        self.mean = 0.0
        self.std = 0.02
        with torch.no_grad():  # No accumulating gradients
            self.embedding.weight[1:].normal_(self.mean, self.std)  # idx=0 is padding so skip

    def forward(self, windows: torch.Tensor,
                features: torch.Tensor | None = None) -> torch.Tensor:

        batch = windows.shape[0]
        x = self.embedding(windows)  # (batch size, count, embedding_dim)
        # Turn to (batch size, count * embedding dim)
        concat = []
        for i in range(x.size(0)):
            flattened = x[i].flatten()
            concat.append(flattened)
        x = torch.stack(concat, dim=0)
        # Add features
        if features is not None:
            x = torch.cat([x, features], dim=1)  # Concatenate embeddings and features
        return self.classifier(x)  # (batch size, num_tags)


def tokenize_word_pieces(tokens: list[str], tokenizer) -> list[list[int]]:
    """Return subword IDs for each word, preserving case and adding no special tokens."""
    res = []
    for token in tokens:
        ids = tokenizer.encode(token, add_special_tokens=False).ids
        res.append(ids)
    return res


def mean_pool_subwords(embeddings: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """Average embeddings [N,S,D] using mask [N,S]; return zero for an empty mask row. Return [N, D]"""
    # Take average of non-masked values [[[D], [D]], [[D], [D] X S] X N]
    # N is words, S is the length fo the longest word, D is dimension
    N, S, D = embeddings.shape
    ret = torch.zeros(N, D, dtype=embeddings.dtype)
    for idx, word_embeds in enumerate(embeddings):  # N times
        word_mask = mask[idx]
        sums = torch.zeros(D, dtype=embeddings.dtype)
        count = 0
        for embed_idx, embed in enumerate(word_embeds):  # S times
            if word_mask[embed_idx] == 0:
                continue  # mask
            # if sum(word_mask) == 0:  # masked
            #     continue
            for i in range(D):
                sums[i] += embed[i]
            count += 1
        for i in range(D):
            if count > 0:
                sums[i] = sums[i] / count
            else:
                sums[i] = 0
        ret[idx] = sums
    return ret


class QwenTagger(nn.Module):
    """Classify frozen word vectors, with an optional shared linear projection."""

    def __init__(self, source_dim: int, projection_dim: int | None = None,
                 radius: int = 2, hidden_dim: int = 128,
                 num_tags: int = 17, dropout: float = 0.1):
        super().__init__()
        self.source_dim = source_dim
        self.projection_dim = projection_dim
        self.radius = radius
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.dropout = dropout
        self.input_dim = 2 * radius + 1
        if self.projection_dim:
            self.classifier = POSMLP(self.input_dim * projection_dim, hidden_dim=hidden_dim, num_tags=num_tags, dropout=dropout)
            self.projection = nn.Linear(source_dim, projection_dim, bias=False)
        else:
            self.classifier = POSMLP(self.input_dim * source_dim, hidden_dim=hidden_dim, num_tags=num_tags, dropout=dropout)
            self.projection = nn.Identity(source_dim)  # projection layer does nothing

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # Windows is [B, 5, d]
        x = self.projection(windows)
        x = x.flatten(1)  # Flatten proj into [B, 5p]
        x = self.classifier(x)
        return x


def first_subword_indices(word_ids: list[int | None],
                          n_words: int) -> list[int]:
    """Return the first subword position for each original word 0..n_words-1.

    None identifies special/padding tokens. Every word must appear in order
    in a contiguous block. Raise ValueError if a word is missing, an ID is
    out of range, or blocks are out of order/noncontiguous. Do not silently
    truncate a word or create a prediction for a special token.

    n_words must be a nonnegative integer; each non-None ID must be an
    integer in [0, n_words). None may not split a word into separate blocks.
    """
    # TODO Task 5
    raise NotImplementedError("Task 5: first_subword_indices")


def extract_bert_words(tokens: list[str], tokenizer, encoder) -> torch.Tensor:
    """Return frozen BERT contextual word representations as float32 CPU
    [N,encoder_hidden_dim] for one sentence.

    tokenizer is a Hugging Face fast tokenizer. Encode with
    is_split_into_words=True, add_special_tokens=True, truncation=False,
    return_tensors='pt'. Reject inputs longer than
    encoder.config.max_position_embeddings. Set encoder.eval(), freeze all
    encoder parameters, use torch.no_grad(), and pass tokenizer model inputs
    (including attention_mask) to the encoder on its device. Select first
    subword positions from outputs.last_hidden_state using the function above.
    Return detached CPU vectors. No pooler_output, mean pooling, or chat text.

    Raise ValueError for overlong inputs, an alignment length different from
    the encoded input length, a missing attention mask, or a selected word
    position masked as padding. Validate word IDs with first_subword_indices.
    """
    # TODO Task 5
    raise NotImplementedError("Task 5: extract_bert_words")


class BertTagger(nn.Module):
    """Classify cached BERT contextual word representations."""

    def __init__(self, source_dim: int = 768, hidden_dim: int = 128,
                 num_tags: int = 17, dropout: float = 0.1):
        super().__init__()
        self.classifier = POSMLP(source_dim, hidden_dim, num_tags, dropout)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.classifier(vectors)


def train_epoch(model: nn.Module, batches, optimizer,
                device: str | torch.device = "cpu") -> dict[str, float]:
    """One epoch: model.train(); for (inputs,labels) batches, transfer tensors
    to device, zero gradients, forward, CE loss, backward, optimizer.step().
    inputs is a tensor dict passed as model(**inputs); labels is long [B].
    Input keys are windows (models 1-4), also features (model 3), or vectors
    (model 5). Model logits have shape [B, len(TAGS)].
    Ignore label -100 in BOTH loss and accuracy. Skip all-ignored batches.
    Return {'loss': token-weighted mean CE, 'accuracy': correct/valid_count}.
    Accuracy is in [0,1]. Raise ValueError if no valid labels in the epoch.
    Do not create/reset the optimizer; it persists across epochs.
    """
    # TODO Task 6
    raise NotImplementedError("Task 6: train_epoch")


def evaluate(model: nn.Module, batches,
             device: str | torch.device = "cpu") -> dict[str, Any]:
    """Set eval mode and use no_grad; perform no parameter updates.

    Return {'loss': token-weighted mean CE, 'accuracy': correct/valid_count,
    'predictions': list[int]} in loader order. predictions includes ONE entry
    per row, even when its gold label is -100; ignored rows do not enter
    metrics. Raise ValueError if there are no valid labels. Same batch format
    as train_epoch. Use argmax logits, with PyTorch's tie handling.
    """
    # TODO Task 6
    raise NotImplementedError("Task 6: evaluate")
