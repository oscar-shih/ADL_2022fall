from typing import List, Dict
import torch
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text, labels, ids = [], [], []
        for sample in samples:
            text.append(sample["text"])
            ids.append(sample["id"])
            if "intent" in sample.keys():
                labels.append(self.label_mapping[sample["intent"]])
        text = torch.LongTensor(self.vocab.encode_batch(text, self.max_len))
        print(labels)
        if len(labels) != 0:
            labels = torch.LongTensor(labels)
        return text, labels, ids

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples: List[Dict]) -> Dict:
        tokens, tags, ids = [], [], []
        for sample in samples:
            tokens.append(sample["tokens"])
            if "tags" in sample.keys():
                tags.append([self.label_mapping[tag] for tag in sample["tags"]])
            ids.append(sample["id"])
        tokens = torch.LongTensor(self.vocab.encode_batch(tokens, self.max_len))
        if len(tags) != 0:
            tags = torch.LongTensor(pad_to_len(tags, self.max_len, 9))
        return tokens, tags, ids
