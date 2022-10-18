# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --rnn_type gru --num_layers 2 --ckpt_dir ./ckpt --test_file ${1} --pred_file ${2}