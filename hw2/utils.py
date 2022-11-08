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

def post_process(ans):
    if "「" in ans and "」" not in ans:
        ans += "」"
    elif "「" not in ans and "」" in ans:
        ans = "「" + ans
    if "《" in ans and "》" not in ans:
        ans += "》"
    elif "《" not in ans and "》" in ans:
        ans = "《" + ans
    ans = ans.replace(",", "")
    return ans

def get_idx(seq, idx=0):
    while seq[idx] != 1:
        idx += 1
    start = idx
    while seq[idx] == 1:
        idx += 1
    end = idx - 1
    return start, end