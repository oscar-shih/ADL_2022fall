def postprocess(preds, labels):
    preds = [pred.strip() + "\n" for pred in preds]
    labels = [label.strip() + "\n" for label in labels]

    return preds, labels