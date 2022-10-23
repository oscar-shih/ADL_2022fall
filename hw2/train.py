import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
import numpy as np
import wandb
import os
from os.path import join
import random
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertConfig, get_cosine_schedule_with_warmup
from argparse import ArgumentParser, Namespace

from dataset import MCDataset, QADataset
from model import MCModel, QAModel
from utils import same_seeds

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(data, model, optimizer, scheduler):
    model.train()
    train_acc, train_loss = [], []


    
    train_acc, train_loss = sum(train_acc) / len(train_acc), sum(train_loss) / len(train_loss)
    return train_acc, train_loss


def validate(data, model):
    model.eval()
    valid_acc, valid_loss = [], []
    with torch.no_grad():
        for batch in tqdm(data):
            _, input_ids, attention_masks, token_type_ids, labels = batch
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
    pass
















if __name__ == "__main__":
    parser = ArgumentParser()
    # Config
    parser.add_argument("--name", type=str, default="Bert-base-uncased")
    parser.add_argument("--seed", type=int, default=7414)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--accu_step", type=int, default=8)
    # Others
    parser.add_argument("--scratch", action="store_true", default=False)


    args = parser.parse_args()
    print(args)
    main(args)