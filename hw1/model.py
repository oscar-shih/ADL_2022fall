from typing import Dict
from pygments import highlight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        rnn_type: str
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.dim = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_type = rnn_type
        self.lstm = nn.LSTM(
            embeddings.size(1), 
            hidden_size, 
            num_layers, 
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.gru = nn.GRU(
            embeddings.size(1),
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.rnn = nn.RNN(
            embeddings.size(1),
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.dim, self.dim),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim, num_class)
        )
        self.slot_classifier = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(0.3),
            nn.Linear(self.dim//2, 9),
        )
        # self.classifier = nn.Sequential(
        #     nn.BatchNorm1d(self.dim),
        #     nn.LeakyReLU(0.25),
        #     nn.Linear(self.dim, num_class)
        # )
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, x) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(x)
        if self.rnn_type == "lstm":
            x, _ = self.lstm(x)
        elif self.rnn_type == "gru":
            x, _ = self.gru(x)
        elif self.rnn_type == "rnn":
            x, _ = self.rnn(x)
        else: 
            raise NotImplementedError
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x

# TODO: Try RNN in Slot Tagging
class SeqTagger(SeqClassifier):

    def forward(self, x) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(x)
        if self.rnn_type == "lstm":
            x, _ = self.lstm(x)
        elif self.rnn_type == "gru":
            x, _ = self.gru(x)
        elif self.rnn_type == "rnn":
            x, _ = self.rnn(x)
        else: 
            raise NotImplementedError
        x = self.slot_classifier(x)
        return x
