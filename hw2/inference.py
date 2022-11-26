import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from dataset import *
from model import *
from utils import post_process, same_seeds
from predict import *

def inference(args):
    mode = "test"
    accelerator = Accelerator(fp16=args.fp16)
    config = AutoConfig.from_pretrained(args.token_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.token_path,
        model_max_length=args.max_len, 
        config=config, 
        use_fast=True
    )
    test_set = MultipleChoiceDataset(
        args, 
        tokenizer, 
        mode=mode
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=1,
        collate_fn=test_set.collate_fn,
        pin_memory=True,
        num_workers=0
    )
    model = MultipleChoiceModel(args, config)
    model.load_state_dict(torch.load(os.path.join(args.mc_ckpt))["model"])
    model, test_loader = accelerator.prepare(model, test_loader)
    rel = mc_predict(model, test_loader, mode)

    del model, test_loader
    torch.cuda.empty_cache()

    test_set = QuestionAnsweringDataset(
        args, 
        tokenizer, 
        mode=mode, 
        relevant=rel
    )
    test_loader = DataLoader(
        test_set,
        collate_fn=test_set.collate_fn,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        num_workers=0
    )

    model = QuestionAnsweringModel(args, config)
    model.load_state_dict(torch.load(os.path.join(args.qa_ckpt))["model"])
    model, test_loader = accelerator.prepare(model, test_loader)

    answers = qa_predict(model, test_loader, mode)
    with open(args.csv_path, "w") as f:
        print("id,answer", file=f)
        for id, answer in answers:
            answer = post_process(answer)
            print(f"{id},{answer}", file=f)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Global
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument("--fp16", type=bool, default=True)
    # File path
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
    )
    parser.add_argument("--context_path", type=Path, default="./data/context.json")
    parser.add_argument("--json_path", type=Path, default="./data/test.json")
    parser.add_argument("--csv_path", type=str, default="./hw2.csv")
    # Model checkpoint
    parser.add_argument("--model_name",type=str, default="hfl/chinese-macbert-base")
    parser.add_argument("--token_path", type=Path, default="./tokenizer")
    parser.add_argument("--mc_ckpt", type=Path, default="./reproduce/mc_2.pt")
    parser.add_argument("--qa_ckpt", type=Path, default="./reproduce/qa_last.pt")

    # Others
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--scratch", type=bool, default=True)
    args = parser.parse_args()
    same_seeds(args.seed)
    inference(args)
