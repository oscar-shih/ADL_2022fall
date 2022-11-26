import os
from argparse import ArgumentParser
from pathlib import Path
import torch
from torch.utils.data import DataLoader
print(torch.cuda.is_available())
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoConfig,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
    Adafactor,
)

from accelerate import Accelerator
import wandb
from tw_rouge import get_rouge
from utils import *
from dataset import SummarizationDataset
from post_process import postprocess
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)
def train(args, model, data, accelerator, tokenizer, optimizer, scheduler):
    train_loss = []
    model.train()
    step = 0
    for text, label in tqdm(data):
        input_ids = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=args.max_src_len,
            return_tensors="pt",
        ).to(device)
        label_input_ids = tokenizer(
            label,
            padding="max_length",
            truncation=True,
            max_length=args.max_tgt_len,
            return_tensors="pt"
        )

        label_input_ids["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in label_input_ids["input_ids"]
        ]
        input_ids["labels"] = torch.LongTensor(label_input_ids["input_ids"]).to(device)

        outputs = model(**input_ids)
        loss = outputs.loss
        loss /= args.accu_step
        train_loss.append(loss.item())
        accelerator.backward(loss)
        if step % args.accu_step == (args.accu_step - 1) or step == (len(data) - 1):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        step += 1
    train_loss = sum(train_loss) / len(train_loss)
    return train_loss

def validate(args, model, tokenizer, data):
    valid_loss = []
    record = {
        "rouge-1":{'r': [], 'p': [], 'f': []},
        "rouge-2":{'r': [], 'p': [], 'f': []},
        "rouge-l":{'r': [], 'p': [], 'f': []},
    }
    model.eval()
    with torch.no_grad():
        for text, label in tqdm(data):
            input_ids = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=args.max_src_len,
                return_tensors="pt"
            ).to(device)
            label_input_ids = tokenizer(
                label,
                padding="max_length",
                truncation=True,
                max_length=args.max_tgt_len,
                return_tensors="pt"
            )
            output = model.generate(**input_ids)
            pred = tokenizer.batch_decode(
                output,
                num_beams=3,
                skip_special_tokens=True
            )
            pred, label = postprocess(pred, label)
            # print(len(prediction) == len(labels))
            record = record_rouge_score(get_rouge(pred, label), record)
            
            label_input_ids["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in label_input_ids["input_ids"]
            ]
            input_ids["labels"] = torch.LongTensor(label_input_ids["input_ids"]).to(device)
            output = model(**input_ids)
            loss = output.loss
            loss /= args.accu_step
            valid_loss.append(loss.item())
        
        valid_loss = sum(valid_loss) / len(valid_loss)
    return valid_loss, record

def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = MT5Tokenizer.from_pretrained(
        args.model_name,
        config=config,
        use_fast=True,
        model_max_length=512
    )
    # model = MT5Model.from_pretrained(
    #     join(args.local_model_root, args.model)
    # )
    model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    train_set = SummarizationDataset(args, "train.jsonl", mode="train")
    valid_set = SummarizationDataset(args, "public.jsonl", mode="valid")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, 
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    optimizer = Adafactor(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.wd,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False, 
    )
    # optimizer = AdamW(
    #     model.parameters(), 
    #     lr=args.lr, 
    #     weight_decay=args.wd
    # )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.batch_size * len(train_loader))
    # scheduler = None
    model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, valid_loader
    )
    wandb_config = {k: v for k, v in vars(args).items()}
    run = wandb.init(
        project=f"ADL Hw3",
        config=wandb_config,
        reinit=True,
        group="Summarization",
        resume="allow"
    )
    artifact = wandb.Artifact("model", type="model")
    best_score = -1
    for ep in range(args.num_epoch):
        train_loss = train(
            args, model, train_loader, accelerator, tokenizer, optimizer, scheduler
        )
        valid_loss, record = validate(
            args, model, tokenizer, valid_loader
        )
        
        valid_r1, valid_r2, valid_rL = [], [], []
        valid_r1 = sum(record["rouge-1"]["f"]) / len(record["rouge-1"]["f"])
        valid_r2 = sum(record["rouge-2"]["f"]) / len(record["rouge-2"]["f"])
        valid_rL = sum(record["rouge-l"]["f"]) / len(record["rouge-l"]["f"])
        mean_score = np.mean([valid_r1, valid_r2, valid_rL])
        print(mean_score)
        wandb.log(
            {
                "Train Loss": train_loss,
                "Validation Loss": valid_loss,
                "rouge r1": valid_r1,
                "rouge r2": valid_r2,
                "rouge rL": valid_rL,
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            torch.save(
                model, 
                os.path.join(args.ckpt_dir, "best.pt")
            )
    torch.save(
        model,
        os.path.join(args.ckpt_dir, "last.pt")
    )

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
        default="./data",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt",
    )
    parser.add_argument("--model_name", type=str, default="google/mt5-small")
    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--accu_step", type=int, default=8)
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--max_tgt_len", type=int, default=64)
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()
    main(args)
