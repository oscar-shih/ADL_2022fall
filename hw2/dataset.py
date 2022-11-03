import json
import os
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MultipleChoiceDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.mode = mode
        self.json_data = []
        try:
            self.json_data = torch.load(os.path.join(args.data_dir, f"mc_{mode}.dat"))
        except Exception as e:
            try:
                path = args.context_path
            except:
                path = os.path.join(args.data_dir, "context.json")
            with open(path, "r") as f:
                context_data = json.load(f)
            try:
                path = args.json_path
            except:
                path = os.path.join(args.data_dir, f"{mode}.json")
            with open(path, "r") as f:
                json_data = json.load(f)
                print(f"Preprocessing {mode} Data:")
                for data in tqdm(json_data):
                    if mode != "test":
                        label = data["paragraphs"].index(data["relevant"])
                    else:
                        label = random.choice(list(range(len(data["paragraphs"]))))

                    qa_pair = [
                        "{} {}".format(data["question"], context_data[i])
                        for i in data["paragraphs"]
                    ]
                    features = tokenizer(
                        qa_pair,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                    self.json_data.append(
                        {
                            "id": data["id"],
                            "input_ids": features["input_ids"],
                            "token_type_ids": features["token_type_ids"],
                            "attention_mask": features["attention_mask"],
                            "label": label,
                        }
                    )
            os.makedirs(args.data_dir, exist_ok=True)
            torch.save(self.json_data, os.path.join(args.data_dir, f"mc_{mode}.dat"))

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        return self.json_data[idx]

    def collate_fn(self, batch):
        ids, input_ids, attention_masks, token_type_ids, labels = [], [], [], [], []
        for sample in batch:
            ids.append(sample["id"])
            input_ids.append(sample["input_ids"])
            token_type_ids.append(sample["token_type_ids"])
            attention_masks.append(sample["attention_mask"])
            labels.append(sample["label"])
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        token_type_ids = torch.stack(token_type_ids)
        labels = torch.LongTensor(labels)
        return ids, input_ids, attention_masks, token_type_ids, labels


class QuestionAnsweringDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", relevant=None):
        assert not (mode == "test" and relevant is None)
        self.mode = mode
        self.tokenizer = tokenizer
        self.json_data = []
        try:
            path = args.context_path
        except:
            path = os.path.join(args.data_dir, "context.json")
        with open(path, "r") as f:
            self.context_data = json.load(f)
        try:
            path = args.json_path
        except:
            path = os.path.join(args.data_dir, f"{mode}.json")
        with open(path, "r") as f:
            json_data = json.load(f)
            print(f"Preprocessing QA {mode} Data:")
            for data in tqdm(json_data):
                tp = {
                    "id": data["id"],
                    "question": data["question"],
                }
                if mode != "test":
                    tp.update(
                        {
                            "context": data["relevant"],
                            "answer": data["answer"],
                        }
                    )
                else:
                    tp.update(
                        {
                            "context": data["paragraphs"][relevant[data["id"]]],
                        }
                    )
                self.json_data.append(tp)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        return self.json_data[idx]

    def collate_fn(self, batch):
        ids = [sample["id"] for sample in batch]
        inputs = self.tokenizer(
            [data["question"] for data in batch],
            [self.context_data[data["context"]] for data in batch],
            truncation="only_second",
            stride=128,  # TODO: change to 32 for model from scatch
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs["context"] = [self.context_data[data["context"]] for data in batch]

        if self.mode != "test":
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                start_char = batch[sample_idx]["answer"]["start"]
                end_char = batch[sample_idx]["answer"]["start"] + len(
                    batch[sample_idx]["answer"]["text"]
                )
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if (
                    offset[context_start][0] > start_char
                    or offset[context_end][1] < end_char
                ):
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = torch.tensor(start_positions)
            inputs["end_positions"] = torch.tensor(end_positions)
        else:
            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []
            offset_mapping = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(batch[sample_idx]["id"])

                sequence_ids = inputs.sequence_ids(i)
                offset = inputs["offset_mapping"][i]
                offset_mapping.append(
                    [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
                )

            inputs["example_id"] = example_ids
            inputs["offset_mapping"] = offset_mapping
        return ids, inputs

