import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os
from tqdm import tqdm
from transformers import MT5Tokenizer, AutoConfig
from accelerate import Accelerator
from pathlib import Path
from utils import same_seeds
from dataset import SummarizationDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
def main(args):
    same_seeds(args.seed)
    test_set = SummarizationDataset(args, mode="test")
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size,
        shuffle=False
    )
    config = AutoConfig.from_pretrained(args.token_path)
    tokenizer = MT5Tokenizer.from_pretrained(
        args.token_path,
        config=config,
        use_fast=True
    )
    accelerator = Accelerator()
    # print(os.path.join(args.ckpt_dir, "best.pt"))
    # model = AutoModel.from_config(config=config)
    model = torch.load("./best.pt")
    model, test_loader = accelerator.prepare(model, test_loader)

    model.eval()
    with torch.no_grad():
        with open(args.output_path, "w", encoding="utf=8") as f:
            for text, id in tqdm(test_loader):
                input_ids = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_src_len,
                    return_tensors="pt"
                ).to(device)
                output = model.generate(**input_ids, do_sample=False, num_beams=10)
                preds = tokenizer.batch_decode(
                    output,
                    skip_special_tokens=True
                )
                preds = [pred.strip() + "\n" for pred in preds]
                for i in range(len(preds)):
                    f.writelines(json.dumps(
                        {
                            'title': preds[i],
                            'id': id[i]
                        }
                    ))
                    f.writelines('\n')
                
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument(
        "--file_path",
        type=Path,
        help="Directory to the dataset.",
        default="./data/public.jsonl",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt",
    )
    parser.add_argument(
        "--token_path",
        type=Path,
        help="Directory to save the tokenizer file.",
        default="./tokenizer",
    )
    parser.add_argument("--model_name", type=str, default="google/mt5-small")

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--max_tgt_len", type=int, default=64)
    parser.add_argument("--output_path", type=str, default="./submission.json")
    args = parser.parse_args()
    main(args)
     