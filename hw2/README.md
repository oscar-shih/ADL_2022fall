# ADL HW 2

## Tasks

1. Context Selection
2. Question Answering

## Context Selection
### Training      
Recommended:
```shell
python3 multiple_choice.py
```

Please use `python3 multiple_choice.py -h` for detailed options.

## Question Answering
### Training 
```shell
python3 question_answering.py
```

## Inference
```shell
python3 inference.py --context_path <context file path> --json_path <input path> --mc_ckpt ./mc.pt --qa_ckpt ./qa.pt --csv_path <output path>
```
