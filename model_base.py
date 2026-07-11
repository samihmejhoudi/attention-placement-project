"""
Shared CNN-LSTM-Attention base architecture.
Three variants differ ONLY in where the attention layer sits.
Everything else (CNN config, LSTM config, classifier head) is identical.
"""

import torch
import torch.nn as nn


class SimpleAttention(nn.Module):
    """A small additive (Bahdanau-style) attention layer.
    Takes a sequence of vectors, returns a single weighted-sum vector."""

    def __init__(self, input_dim):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        weights = torch.softmax(self.score(x), dim=1)   # (batch, seq_len, 1)
        context = torch.sum(weights * x, dim=1)          # (batch, input_dim)
        return context, weights


class CNNLSTMAttention(nn.Module):
    """
    One shared architecture. `attention_position` controls WHERE
    the SimpleAttention layer is inserted:
        - "before_cnn"   : Embedding -> Attention -> CNN -> LSTM -> Classifier
        - "between"      : Embedding -> CNN -> Attention -> LSTM -> Classifier
        - "after_lstm"   : Embedding -> CNN -> LSTM -> Attention -> Classifier
    """

    def __init__(self, vocab_size, embed_dim=128, cnn_channels=64,
                 lstm_hidden=128, num_classes=4, attention_position="after_lstm"):
        super().__init__()
        assert attention_position in ("before_cnn", "between", "after_lstm")
        self.attention_position = attention_position

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # CNN expects (batch, channels, seq_len) -> Conv1d over embed_dim channels
        self.cnn = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_channels,
                              kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden,
                             batch_first=True)

        # attention operates on whichever dimension it sees at its position
        if attention_position == "before_cnn":
            self.attention = SimpleAttention(embed_dim)
            self.classifier = nn.Linear(lstm_hidden, num_classes)
        elif attention_position == "between":
            self.attention = SimpleAttention(cnn_channels)
            self.classifier = nn.Linear(lstm_hidden, num_classes)
        else:  # after_lstm
            self.attention = SimpleAttention(lstm_hidden)
            self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) token ids
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)

        if self.attention_position == "before_cnn":
            context, _ = self.attention(emb)
            # broadcast context back across sequence before CNN (simple approach:
            # use attention-weighted embedding sequence, not just the pooled vector)
            weights = torch.softmax(self.attention.score(emb), dim=1)
            emb = emb * weights  # reweight tokens, keep sequence shape
            cnn_out = self.relu(self.cnn(emb.transpose(1, 2))).transpose(1, 2)
            lstm_out, (h_n, _) = self.lstm(cnn_out)
            final = h_n[-1]

        elif self.attention_position == "between":
            cnn_out = self.relu(self.cnn(emb.transpose(1, 2))).transpose(1, 2)
            weights = torch.softmax(self.attention.score(cnn_out), dim=1)
            cnn_out = cnn_out * weights
            lstm_out, (h_n, _) = self.lstm(cnn_out)
            final = h_n[-1]

        else:  # after_lstm
            cnn_out = self.relu(self.cnn(emb.transpose(1, 2))).transpose(1, 2)
            lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, lstm_hidden)
            final, _ = self.attention(lstm_out)  # pooled context vector

        logits = self.classifier(final)
        return logits


if __name__ == "__main__":
    # Quick sanity test: fake batch of token ids, run all 3 variants
    vocab_size = 1000
    batch_size = 8
    seq_len = 40
    fake_input = torch.randint(1, vocab_size, (batch_size, seq_len))

    for pos in ["before_cnn", "between", "after_lstm"]:
        model = CNNLSTMAttention(vocab_size=vocab_size, attention_position=pos)
        out = model(fake_input)
        print(f"[{pos}] output shape: {out.shape}  (expected: [{batch_size}, 4])")