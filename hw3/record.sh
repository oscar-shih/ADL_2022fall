exp=1
for file in greedy.jsonl nonegreedy.jsonl beam3.jsonl beam5.jsonl beam10.jsonl k50.jsonl k100.jsonl p03.jsonl p05.jsonl p07.jsonl t01.jsonl t03.jsonl t05.jsonl t07.jsonl 
do
    echo $file
    python3 eval.py -r ./data/public.jsonl -s $file -o $exp.json
    val=`expr $exp + 1`
    exp=$val
done