import torch
import torch.nn as nn
from transformers import AutoModelForMultipleChoice, AutoModelForQuestionAnswering

class MultipleChoiceModel(nn.Module):
    def __init__(self, args, config):
        super(MultipleChoiceModel, self).__init__()
        self.name = args.model_name
        if args.scratch: # For experiment in report
            self.model = AutoModelForMultipleChoice.from_config(config)
        else:
            self.model = AutoModelForMultipleChoice.from_pretrained(self.name, config=config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class QuestionAnsweringModel(nn.Module):
    def __init__(self, args, config):
        super(QuestionAnsweringModel, self).__init__()
        self.name = args.model_name
        if args.scratch: # For experiment in report
            self.model = AutoModelForQuestionAnswering.from_config(config)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.name, config=config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
