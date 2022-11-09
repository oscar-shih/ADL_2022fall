import os
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import get_idx
from preprocess import mc_preprocess, qa_preprocess

class MultipleChoiceDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.mode = mode # "train", "valid" or "test"
        self.data = []
        try:
            self.data = torch.load(os.path.join(args.data_dir, f"mc_{mode}.dat"))
        except Exception as e:
            context_path = os.path.join(args.data_dir, "context.json")
            with open(context_path, "r") as f:
                context_data = json.load(f)
            path = os.path.join(args.data_dir, f"{self.mode}.json")
            with open(path, "r") as f:
                data = json.load(f)
                for d in tqdm(data):
                    ids, feature, label = mc_preprocess(
                        context=context_data,
                        data=d,
                        tokenizer=tokenizer,
                        mode=self.mode
                    )
                    self.data.append(
                        {
                            "id": ids,
                            "input_ids": feature["input_ids"],
                            "token_type_ids": feature["token_type_ids"],
                            "attention_mask": feature["attention_mask"],
                            "label": label,
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        ids, input_ids, attention_masks, token_type_ids, labels = [], [], [], [], []
        for sample in batch:
            ids.append(sample["id"])
            input_ids.append(sample["input_ids"])
            attention_masks.append(sample["attention_mask"])
            token_type_ids.append(sample["token_type_ids"])     
            labels.append(sample["label"])
        labels = torch.LongTensor(labels)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        token_type_ids = torch.stack(token_type_ids)
        if self.mode != "test":
            return input_ids, attention_masks, token_type_ids, labels
        else: # For inference
            return ids, input_ids, attention_masks, token_type_ids

class QuestionAnsweringDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", relevant=None):
        self.mode = mode
        self.tokenizer = tokenizer
        self.data = []
        context_path = os.path.join(args.data_dir, "context.json")
        with open(context_path, "r") as f:
            self.context_data = json.load(f)
        path = os.path.join(args.data_dir, f"{mode}.json")
        with open(path, "r") as f:
            data = json.load(f)
            for d in tqdm(data):
                qa = qa_preprocess(
                    data=d, 
                    relevant=relevant, 
                    mode=self.mode
                )
                self.data.append(qa)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        ids = [sample["id"] for sample in batch]
        inputs = self.tokenizer(
            [data["question"] for data in batch],
            [self.context_data[data["context"]] for data in batch],
            truncation="only_second",
            stride=32,  
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs["context"] = [self.context_data[data["context"]] for data in batch]

        if "answer" in self.data[0].keys():
            sample_map = inputs.pop("overflow_to_sample_mapping")
            offset_mapping = inputs.pop("offset_mapping")
            start_positions, end_positions = [], []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                start_char, end_char = batch[sample_idx]["answer"]["start"], batch[sample_idx]["answer"]["start"] + len(batch[sample_idx]["answer"]["text"])
                seq_id = inputs.sequence_ids(i)
                start, end = get_idx(seq_id)

                if offset[start][0] > start_char or offset[end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    idx = start
                    while offset[idx][0] <= start_char and idx <= end :
                        idx += 1
                    start_positions.append(idx - 1)
                    idx = end
                    while offset[idx][1] >= end_char and idx >= start:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = torch.LongTensor(start_positions)
            inputs["end_positions"] = torch.LongTensor(end_positions)
        else:
            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids, offset_mapping = [], []

            for id in range(len(inputs["input_ids"])):
                sample_idx = sample_map[id]
                example_ids.append(batch[sample_idx]["id"])

                seq_id = inputs.sequence_ids(id)
                offset = inputs["offset_mapping"][id]
                offset_mapping.append(
                    [v if seq_id[k] else None for k, v in enumerate(offset)]
                )

            inputs["example_id"] = example_ids
            inputs["offset_mapping"] = offset_mapping
        return ids, inputs