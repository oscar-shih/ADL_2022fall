import os
from argparse import ArgumentParser, Namespace
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

log_step = 1


def train(accelerator, data_loader, model, optimizer, scheduler=None):
    train_loss = []
    train_accs = []

    model.train()

    for idx, batch in enumerate(tqdm(data_loader)):
        ids, inputs = batch
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
        loss = loss / 4
        accelerator.backward(loss)

        start_logits = qa_output.start_logits.argmax(dim=-1)
        end_logits = qa_output.end_logits.argmax(dim=-1)
        acc = (
            ((start_positions == start_logits) & (end_positions == end_logits))
            .cpu()
            .numpy()
            .mean()
        )

        if ((idx + 1) % 4 == 0) or (idx == len(data_loader) - 1):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    return train_loss, train_acc


@torch.no_grad()
def validate(data_loader, model):
    model.eval()
    valid_loss = []
    valid_accs = []

    # log_step = 1
    for batch in tqdm(data_loader):
        ids, inputs = batch
        n = inputs["input_ids"].shape[0]
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

        start_logits = qa_output.start_logits.argmax(dim=-1)
        end_logits = qa_output.end_logits.argmax(dim=-1)
        acc = (
            ((start_positions == start_logits) & (end_positions == end_logits))
            .cpu()
            .numpy()
            .mean()
        )

        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    return valid_loss, valid_acc


def main(args):
    same_seeds(args.seed)
    accelerator = Accelerator(fp16=True)
    # print(f"Using {accelerator.device}")

    if args.scratch:
        config = BertConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            classifier_dropout=0.4,
            pooler_fc_size=512,
            pooler_num_attention_heads=4,
        )
    else:
        config = AutoConfig.from_pretrained(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, config=config, model_max_length=args.max_len, use_fast=True
    )
    model = QuestionAnsweringModel(args, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    starting_epoch = 1
    if args.load:
        print(f"loading model from {args.load}")
        ckpt = torch.load(args.load)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        starting_epoch = ckpt["epoch"]
    wandb_config = {k: v for k, v in vars(args).items()}
    run = wandb.init(
        project=f"ADL Hw2",
        config=wandb_config,
        reinit=True,
        group="Question Answering",
        resume="allow"
    )
    artifact = wandb.Artifact("model", type="model")

    train_set = QuestionAnsweringDataset(args, tokenizer)
    valid_set = QuestionAnsweringDataset(args, tokenizer, mode="valid")

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

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 0, args.num_epoch * len(train_loader) // args.accu_step + args.num_epoch
    )

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    best_loss = float("inf")

    for epoch in range(starting_epoch, args.num_epoch + 1):
        train_loss, train_acc = train(
            accelerator, train_loader, model, optimizer, scheduler
        )
        print(f"Train Accuracy: {train_acc:.2f}, Train Loss: {train_loss:.2f}")
        valid_loss, valid_acc = validate(valid_loader, model)
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
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(args.ckpt_dir, f"best_qa.pt"),
            )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(args.ckpt_dir, "qa_last.pt"),
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

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--accu_step", type=int, default=4)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--scratch", action="store_true")

    args = parser.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)