import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    BertConfig,
    get_cosine_schedule_with_warmup,
)
from dataset import QuestionAnsweringDataset
from model import QuestionAnsweringModel
from utils import same_seeds

def train(data, model, optimizer, scheduler, accelerator):
    model.train()
    train_acc, train_loss = [], []

    for idx, batch in enumerate(tqdm(data)):
        _, inputs = batch
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]
        qa_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        loss = qa_output.loss
        loss /= 4
        accelerator.backward(loss)
        train_loss.append(loss.item())

        start_logits = qa_output.start_logits.argmax(dim=-1)
        end_logits = qa_output.end_logits.argmax(dim=-1)
        acc = (
            ((start_positions == start_logits) & (end_positions == end_logits)).cpu().numpy().mean()
        )
        train_acc.append(acc)

        if ((idx + 1) % 4 == 0) or (idx == len(data) - 1):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    
    train_acc, train_loss = sum(train_acc) / len(train_acc), sum(train_loss) / len(train_loss)
    return train_acc, train_loss

def validate(data_loader, model):
    model.eval()
    valid_acc, valid_loss = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            _, inputs = batch
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            start_positions = inputs["start_positions"]
            end_positions = inputs["end_positions"]
            qa_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = qa_output.loss
            valid_loss.append(loss.item())

            start_logits = qa_output.start_logits.argmax(dim=-1)
            end_logits = qa_output.end_logits.argmax(dim=-1)
            acc = (
                ((start_positions == start_logits) & (end_positions == end_logits)).cpu().numpy().mean()
            )
            valid_acc.append(acc)

        valid_acc, valid_loss = sum(valid_acc) / len(valid_acc), sum(valid_loss) / len(valid_loss)
    return valid_acc, valid_loss

def get_dataloader_qa(args, tokenizer, mode):
    if mode == "valid":
        dataset = QuestionAnsweringDataset(args, tokenizer, mode)
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            batch_size=args.batch_size,
        )
    else:
        dataset = QuestionAnsweringDataset(args, tokenizer)
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            batch_size=args.batch_size,
        )

    return dataloader

def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)
    wandb_config = {k: v for k, v in vars(args).items()}
    run = wandb.init(
        project=f"ADL Hw2",
        config=wandb_config,
        reinit=True,
        group="Question Answering",
        resume="allow"
    )
    artifact = wandb.Artifact("model", type="model")
    if args.scratch:
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
        config = AutoConfig.from_pretrained(args.model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, config=config, model_max_length=args.max_len, use_fast=True
    )

    train_loader = get_dataloader_qa(args, tokenizer, mode="train")
    valid_loader= get_dataloader_qa(args, tokenizer, mode="valid")

    model = QuestionAnsweringModel(args, config)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0, args.num_epoch * len(train_loader)
    )

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    best_acc = 0
    for ep in range(args.num_epoch):
        train_acc, train_loss = train(train_loader, model, optimizer, scheduler, accelerator)
        valid_acc, valid_loss = validate(valid_loader, model)

        wandb.log(
            {
                "Train Accuracy": train_acc,
                "Train Loss": train_loss,
                "Validation Accuracy": valid_acc,
                "Validation Loss": valid_loss,
            }
        )
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.ckpt_dir, f"{args.model_name}_best_qa.pt"),
            )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(args.ckpt_dir, f"{args.model_name}_qa_last.pt"),
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
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base")
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt",
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=1e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epoch", type=int, default=6)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--scratch", action="store_true")

    args = parser.parse_args()
    main(args)