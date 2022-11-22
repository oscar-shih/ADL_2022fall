import torch
import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def record_rouge_score(score, record):
    for key in score.keys():
        for idx in score[key].keys():
            record[key][idx].append(score[key][idx])
    return record