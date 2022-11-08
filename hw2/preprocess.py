import torch
import random
from tqdm import tqdm

def mc_preprocess(context, data, tokenizer, mode):
    ids = data["id"]
    lbl = data["paragraphs"].index(data["relevant"]) if mode != "test" else random.choice(list(range(len(data["paragraphs"]))))
    pair = [
        "{} {}".format(data["question"], context[i]) for i in data["paragraphs"]
    ] # May Change to [SEP]
    feature = tokenizer(
        pair,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return ids, feature, lbl

def qa_preprocess(data, relevant, mode):
    qa = {
        "id": data["id"],
        "question": data["question"]
    }
    qa.update({
        "answer": data["answer"],
        "context": data["relevant"]
    }) if mode != "test" else qa.update({"context": data["paragraphs"][relevant[data["id"]]]})
    return qa
