# ADL_fall

## Download model for testing
```bash=
bash download.sh
```

## Test Intent Classification
```bash=
bash intent_cls.sh $1 $2
```

## Test Slot Tagging
```bash=
bash slot_tag.sh $1 $2
```

## Reproduce Training Intent Classification
```bash=
python3 train_intent.py --rnn_type gru --num_layers 2
```

## Reproduce Training Intent Classification
```bash=
 python3 train_slot.py --rnn_type gru --num_layers 2   
```