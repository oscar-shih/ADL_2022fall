from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset
import os
import json
import pandas

class SummarizationDataset(Dataset):
    def __init__(self, args, file, mode):
        super(SummarizationDataset, self).__init__()
        self.id = []
        self.data = {}
        self.label = {}
        self.mode = mode
        with open(os.path.join(args.data_dir, file)) as f:
            if self.mode != "test":
                for line in f:
                    line = json.loads(line)
                    self.id.append(line['id'])
                    self.data[line['id']] = line['maintext'].strip()
                    self.label[line['id']] = line['title'].strip()
                    
            else:
                for line in f:
                    line = json.loads(line) # dict_keys(['date_publish', 'source_domain', 'maintext', 'split', 'id'])
                    self.id.append(line['id'])
                    self.data[line['id']] = line['maintext'].strip()
        # with open(f"./data/{mode}.json", "w", encoding="utf-8") as f:
        #     json.dump(self.data, f, indent=4, ensure_ascii=False)            

    def __getitem__(self, index):
        if self.mode == "test":
            return self.data[self.id[index]], self.id[index]
        else:
            return self.data[self.id[index]], self.label[self.id[index]]

    def __len__(self):
        return len(self.id)
        
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--data_dir", type=str, default="./data")
#     args = parser.parse_args()
#     set = SummarizationDataset(args, file="./train.jsonl", mode="test")
