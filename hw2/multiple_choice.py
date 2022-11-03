import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertConfig, get_cosine_schedule_with_warmup

from dataset import MultipleChoiceDataset
from model import MultipleChoiceModel
from utils import same_seeds


def train(data, model, optimizer, scheduler, accelerator):
    train_accs, train_loss = [], []
    model.train()

    for idx, batch in enumerate(tqdm(data)):
        _, input_ids, attention_masks, token_type_ids, labels = batch
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
        loss = loss / 8
        accelerator.backward(loss)
        train_accs.append(acc)
        train_loss.append(loss.item())
        
        if (idx + 1) % 8 == 0 or idx == len(data) - 1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    train_acc, train_loss = sum(train_accs) / len(train_accs),sum(train_loss) / len(train_loss)

    return train_acc, train_loss

def validate(data, model):
    model.eval()
    valid_loss = []
    valid_accs = []
    with torch.no_grad():
        for batch in tqdm(data):
            _, input_ids, attention_masks, token_type_ids, labels = batch
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

    return valid_acc, valid_loss

def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)
    if args.scratch:
        config = BertConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            classifier_dropout=0.4,
            pooler_fc_size=256,
            pooler_num_attention_heads=4,
            return_dict=False,
        )
    else:
        config = AutoConfig.from_pretrained(args.model_name, return_dict=False)
    wandb_config = {k: v for k, v in vars(args).items()}
    run = wandb.init(
        project=f"ADL Hw2",
        config=wandb_config,
        reinit=True,
        group="Multiple Choise",
        resume="allow"
    )
    artifact = wandb.Artifact("model", type="model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, config=config, model_max_length=args.max_len, use_fast=True
    )
    model = MultipleChoiceModel(args, config)
    train_set = MultipleChoiceDataset(args, tokenizer)
    valid_set = MultipleChoiceDataset(args, tokenizer, mode="valid")

    train_loader = DataLoader(
        train_set,
        collate_fn=train_set.collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
    )
    valid_loader = DataLoader(
        valid_set,
        collate_fn=valid_set.collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
    )
    warmup_step = int(0.1 * len(train_loader)) // args.accu_step
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_step, args.num_epoch * len(train_loader)
    )

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )
    best_loss = float("inf")
    for epoch in range(1, args.num_epoch + 1):
        train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, accelerator
        )
        valid_acc, valid_loss = validate(valid_loader, model)
        print(f"Train Accuracy: {train_acc:.2f}, Train Loss: {train_loss:.2f}")
        print(f"Valid Accuracy: {valid_acc:.2f}, Valid Loss: {valid_loss:.2f}")
        wandb.log(
            {
                "Train Accuracy": train_acc,
                "Train Loss": train_loss,
                "Validation Accuracy": valid_acc,
                "Validation Loss": valid_loss,
            }
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.ckpt_dir, f"mc_{epoch}.pt"),
            )
    torch.save(
        {"model": model.state_dict(),
         "optimizer": optimizer.state_dict(),
        },
        os.path.join(args.ckpt_dir, "last.pt"),
    )
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base")
    # data
    parser.add_argument("--max_len", type=int, default=512)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=8)
    parser.add_argument("--accu_step", type=int, default=8)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--scratch", action="store_true")

    args = parser.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
