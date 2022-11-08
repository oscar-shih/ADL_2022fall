# for model in roberta-base albert-base-v2 bert-base-uncased
# do
#     python3 multiple_choice.py --model_name $model
#     python3 question_answering.py --model_name $model
# done

# for model in roberta-base albert-base-v2 bert-base-uncased
model=$1
python3 multiple_choice.py --model_name $model --scratch --batch_size 8 --num_epoch 15
python3 question_answering.py --model_name $model --scratch --batch_size 8 --num_epoch 15
