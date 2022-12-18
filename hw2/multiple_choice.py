import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertConfig, get_cosine_schedule_with_warmup

from dataset import MultipleChoiceDataset
from model import MultipleChoiceModel
from utils import same_seeds

def train(data, model, optimizer, scheduler, accelerator):
    model.train()
    train_acc, train_loss = [], []

    for i, batch in enumerate(tqdm(data)):
        input_ids, attention_masks, token_type_ids, labels = batch
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
            labels=labels
        )
        loss /= 8 # Accumulation steps = 8
        accelerator.backward(loss)
        train_loss.append(loss.item())

        acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
        train_acc.append(acc)
        
        if (i + 1) % 8 == 0 or i == (len(data) - 1):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    train_acc, train_loss = sum(train_acc) / len(train_acc),sum(train_loss) / len(train_loss)
    return train_acc, train_loss

def validate(data, model):
    model.eval()
    valid_acc, valid_loss = [], []
    
    with torch.no_grad():
        for batch in tqdm(data):
            input_ids, attention_masks, token_type_ids, labels = batch
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids,
                labels=labels
            )
            valid_loss.append(loss.item())

            acc = (logits.argmax(dim=-1) == labels).cpu().float().mean()
            valid_acc.append(acc)

        valid_acc, valid_loss = sum(valid_acc) / len(valid_acc), sum(valid_loss) / len(valid_loss)
    return valid_acc, valid_loss

def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)
    if args.scratch: # For experiment in report
        config = BertConfig(
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            classifier_dropout=0.3,
            pooler_fc_size=256,
            pooler_num_attention_heads=4,
            return_dict=False
        )
    else:
        config = AutoConfig.from_pretrained(args.model_name, return_dict=False)
    # print(config)
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
        args.model_name, 
        config=config, 
        model_max_length=args.max_len, 
        use_fast=True
    )
    model = MultipleChoiceModel(args, config)
    train_set = MultipleChoiceDataset(args, tokenizer)
    valid_set = MultipleChoiceDataset(args, tokenizer, mode="valid")

    train_loader = DataLoader(
        train_set,
        collate_fn=train_set.collate_fn,
        shuffle=True,
        batch_size=args.batch_size
    )
    valid_loader = DataLoader(
        valid_set,
        collate_fn=valid_set.collate_fn,
        shuffle=False,
        batch_size=args.batch_size
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0, args.num_epoch * len(train_loader)
    )

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )
    best_acc = 0
    for ep in range(args.num_epoch):
        train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, accelerator
        )
        valid_acc, valid_loss = validate(valid_loader, model)
        wandb.log(
            {
                "Train Accuracy": train_acc,
                "Train Loss": train_loss,
                "Valid Accuracy": valid_acc,
                "Valid Loss": valid_loss
            }
        )
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(
                {
                 "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.ckpt_dir, f"{args.model_name}_mc_best.pt"),
            )
    torch.save(
        {
         "model": model.state_dict(),
         "optimizer": optimizer.state_dict(),
        },
        os.path.join(args.ckpt_dir, f"{args.model_name}_mc_last.pt"),
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
        help="Directory to save models.",
        default="./ckpt",
    )
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base")
    # data
    parser.add_argument("--max_len", type=int, default=512)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=6)

    # training settings
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--scratch", action="store_true")

    args = parser.parse_args()
    main(args)
