# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# python3 test_intent.py --rnn_type lstm --num_layers 2 --test_file data/intent/test.json --ckpt_path ckpt/intent
python3 test_intent.py --rnn_type gru --num_layers 2 --test_file "${1}" --ckpt_path ./ckpt --pred_file "${2}"
