# ADL HW 2

## Tasks

1. Context Selection
2. Question Answering

## Checkpoint download
```shell
bash download.sh
```

## Context Selection
### Training      
```shell
python3 multiple_choice.py
```

## Question Answering
### Training 
```shell
python3 question_answering.py
```

## Inference
```shell
python3 inference.py --context_path <context file path> --json_path <input path> --csv_path <output path> --mc_ckpt ./mc.pt --qa_ckpt ./qa.pt 
```
