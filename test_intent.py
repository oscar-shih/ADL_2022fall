import json
from mimetypes import suffix_map
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from os.path import join
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab, same_seeds

same_seeds(1126)
device = "cuda" if torch.cuda.is_available() else "cpu"
def inference(model, data):
    ids, pred = [], []
    model.eval()
    with torch.no_grad():
        for text, _, id in tqdm(data):
            text = text.to(device)
            logits = model(text)
            pred.extend(logits.argmax(dim=-1).cpu().numpy())
            ids.extend(id)
        return pred, ids

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    idx2intent = {k: v for k, v in enumerate(intent2idx)}
    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        rnn_type=args.rnn_type
    ).to(device)
    
    model_path = "intent.pt"
    ckpt = torch.load(join(args.ckpt_path, f"{args.num_layers}-{args.rnn_type}", model_path))
    model.load_state_dict(ckpt["model"])
    model.eval()
    # load weights into model

    # TODO: predict dataset
    pred, ids = inference(model, test_loader)
    os.makedirs(join(args.rnn_type, f"{args.num_layers}-{args.rnn_type}"), exist_ok=True)
    with open(join(args.rnn_type, f"{args.num_layers}-{args.rnn_type}", args.pred_file), "w") as out:
        out.write("id,intent\n")
        for p, id in zip(pred, ids):
            out.write(id + "," + idx2intent[p] + "\n")

    # TODO: write prediction to file (args.pred_file)


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
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_type", type=str, default="rnn")
    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--best", type=bool, default=True)
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
