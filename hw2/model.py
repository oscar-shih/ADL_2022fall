import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMultipleChoice, AutoModelForQuestionAnswering

class MCModel(nn.Module):
    def __init__(self, args, config) -> None:
        super(MCModel, self).__init__()
        self.name = args.name
        if args.scratch:
            self.model = AutoModelForMultipleChoice.from_config(config)
        self.model = AutoModelForMultipleChoice.from_pretrained(self.name, config)
    
    def forward(self, *args, **kwargs):
        self.model(*args, **kwargs)
    

class QAModel(nn.Module):
    def __init__(self, args, config) -> None:
        super(QAModel, self).__init__()
        self.name = args.name
        if args.scratch:
            self.model = AutoModelForQuestionAnswering.from_config(config)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.name, config)

    def forward(self, *args, **kwargs):
        self.model(*args, **kwargs)