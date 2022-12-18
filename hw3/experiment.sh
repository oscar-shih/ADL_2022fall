python3 submission.py --greedy False --output_path ./greedy.jsonl
python3 submission.py --greedy True --output_path ./nonegreedy.jsonl


python3 submission.py --beams 3 --output_path ./beam3.jsonl
python3 submission.py --beams 5 --output_path ./beam5.jsonl
python3 submission.py --beams 10 --output_path ./beam10.jsonl

python3 submission.py --top_k 50 --output_path ./k50.jsonl
python3 submission.py --top_k 100 --output_path ./k100.jsonl
python3 submission.py --top_k 200 --output_path ./k200.jsonl

python3 submission.py --top_p 0.3 --output_path ./p03.jsonl
python3 submission.py --top_p 0.5 --output_path ./p05.jsonl
python3 submission.py --top_p 0.7 --output_path ./p07.jsonl


python3 submission.py --temperature 0.1 --output_path ./t01.jsonl
python3 submission.py --temperature 0.3 --output_path ./t03.jsonl
python3 submission.py --temperature 0.5 --output_path ./t05.jsonl
python3 submission.py --temperature 0.7 --output_path ./t07.jsonl