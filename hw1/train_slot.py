import os
from os.path import join
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from sched import scheduler
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab, same_seeds, get_cosine_schedule_with_warmup

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, data, optimizer):
    model.train()
    train_acc, train_loss = [], []
    criterion = nn.CrossEntropyLoss()
    for tokens, tags, _ in tqdm(data):
        tokens = tokens.to(device)
        tags = tags.to(device)

        logits = model(tokens)
        loss = criterion(logits.transpose(1, 2), tags)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = ((logits.argmax(dim=-1) == tags).sum(dim=1)== args.max_len).cpu().int().numpy()        
        train_acc.extend(acc)

    train_acc, train_loss = sum(train_acc) / len(train_acc), sum(train_loss) / len(train_loss)
    return train_acc, train_loss

def validate(model, data):
    model.eval()
    dev_acc, dev_loss = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for tokens, tags, _ in tqdm(data):
            tokens, tags = tokens.to(device), tags.to(device)
            logits = model(tokens)

            loss = criterion(logits.transpose(1, 2), tags)
            dev_loss.append(loss.item())
            
            acc = ((logits.argmax(dim=-1) == tags).sum(dim=1) == args.max_len).cpu().int().numpy()
            dev_acc.extend(acc)

    dev_acc = sum(dev_acc) / len(dev_acc)
    dev_loss = sum(dev_loss) / len(dev_loss)
    return dev_acc, dev_loss

def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = DataLoader(datasets[TRAIN], shuffle=True, batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn)
    dev_loader = DataLoader(datasets[DEV], shuffle=False, batch_size=args.batch_size, collate_fn=datasets[DEV].collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqTagger(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=10,
        rnn_type=args.rnn_type
    ).to(device)
    print(model)
    wandb_config = {k: v for k, v in vars(args).items()}
    run = wandb.init(
        project=f"ADL Hw1",
        config=wandb_config,
        reinit=True,
        group="Sequence Tagging",
        resume="allow"
    )
    artifact = wandb.Artifact("model", type="model")
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.99), weight_decay=1e-5, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.num_epoch)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = -1
    os.makedirs(join(args.ckpt_dir, f"{args.num_layers}-{args.rnn_type}"), exist_ok=True)
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        train_acc, train_loss = train(model, train_loader, optimizer)
        dev_acc, dev_loss = validate(model, dev_loader)
        wandb.log(
            {'Train Acc': train_acc,
             'Train Loss': train_loss,
             'Dev Acc': dev_acc,
             'Dev Loss': dev_loss}
        )
        if scheduler is not None:
            scheduler.step()
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
                        join(args.ckpt_dir, f"{args.num_layers}-{args.rnn_type}", "slot.pt"))
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()},
                join(args.ckpt_dir, f"{args.num_layers}-{args.rnn_type}", "last.pt"))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_type", type=str, default="lstm")
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    same_seeds(1126)
    main(args)