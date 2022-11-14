import nltk
def postprocess(preds, labels):
    preds = [pred.strip().replace("<extra_id_0>", "") for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    preds = [
        "NOT FOUND\n" if (len(pred) <= 0 or pred == ".") else pred for pred in preds
    ]
    return preds, labels