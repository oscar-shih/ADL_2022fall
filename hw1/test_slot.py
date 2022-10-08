from hashlib import new
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab, same_seeds

same_seeds(1126)
device = "cuda" if torch.cuda.is_available() else "cpu"
def inference(model, data):
    ids, pred = [], []
    model.eval()
    with torch.no_grad():
        for tokens, _, id in tqdm(data):
            tokens = tokens.to(device)
            logits = model(tokens)
            pred.extend(logits.argmax(dim=-1))
            ids.extend(id)
        return pred, ids

def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tags_idx_path = args.cache_dir / "tag2idx.json"
    tags2idx: Dict[str, int] = json.loads(tags_idx_path.read_text())
    idx2tags = {k: v for k, v in enumerate(tags2idx)}

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tags2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn)

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
    model_path = "slot.pt"
    ckpt = torch.load(join(args.ckpt_dir, f"{args.num_layers}-{args.rnn_type}", model_path))
    model.load_state_dict(ckpt["model"])
    model.eval()
    pred, ids = inference(model, test_loader)


    os.makedirs(join(args.rnn_type, f"{args.num_layers}-{args.rnn_type}"), exist_ok=True)
    with open(join(args.rnn_type, f"{args.num_layers}-{args.rnn_type}", args.pred_file), "w") as out:
        out.write("id,tags\n")
        new_pred = []
        for p, id in zip(pred, ids):
            for i in range(len(p)):
                if p[i] == 9:
                    idx = i
                    break
            new_pred = p[:idx]
            out.write(id + ",")
            for i in range(len(new_pred)-1):
                p = new_pred[i].item()
                out.write(idx2tags[p] + " ")
            out.write(idx2tags[new_pred[-1].item()]+ "\n")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_type", type=str, default="lstm")
    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)