import json
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

with open("./data/slot/eval.json", "r") as f:
    dataset = json.load(f)

y_true = [data["tags"] for data in dataset]

with open("./slot.csv", "r") as f:
    y_pred = []
    for l in f.readlines()[1:]:
        y_pred.append(l.split(',')[1].split())
print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))