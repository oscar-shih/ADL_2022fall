import os
import numpy as np
import torch
from tqdm import tqdm

def mc_predict(data, model):
    model.eval()
    relevant = dict()
    with torch.no_grad():
        for batch in tqdm(data):
            ids, input_ids, attention_masks, token_type_ids = batch
            output = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids
            )
            pred = output.logits.argmax(dim=-1).cpu().numpy()
            for id, p in zip(ids, pred):
                relevant[id] = int(p)

    return relevant

def qa_predict(data, model): 
    model.eval()
    ans = []
    with torch.no_grad():
        for batch in tqdm(data):
            answers = []
            ids, inputs = batch
            context = inputs["context"][0]
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]

            qa_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            start_logits = qa_output.start_logits.cpu().numpy()
            end_logits = qa_output.end_logits.cpu().numpy()
            for i in range(len(input_ids)):
                start_logit = start_logits[i]
                end_logit = end_logits[i]
                offsets = inputs["offset_mapping"][i]

                start_indexes = np.argsort(start_logit)[-1:-21:-1].tolist()
                end_indexes = np.argsort(end_logit)[-1:-21:-1].tolist()

                for start in start_indexes:
                    for end in end_indexes:
                        if offsets[start] is None or offsets[end] is None or end < start:
                            continue
                        answers.append(
                            {
                                "text": context[offsets[start][0]:offsets[end][1]],
                                "logit_score": start_logit[start] + end_logit[end]
                            }
                        )
            best_answer = max(answers, key=lambda x: x["logit_score"])
            ans.append((ids[0], best_answer["text"]))
    return ans