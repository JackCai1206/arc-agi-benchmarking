# export OPENAI_BASE_URL=http://0.0.0.0:8000/v1
# python main.py --data_dir data/arc-agi/data/evaluation --config qwen3-8b --task_id 0a1d4ef5 --log-level DEBUG --enable-metrics --verbose

MODEL=o3-2025-04-16-low-2025-06-10
# MODEL=gpt-4o-2024-11-20
SPLIT=evaluation
# python main.py --data_dir data/arc-agi/data/$SPLIT --config $MODEL --task_id 009d5c81 --log-level DEBUG --enable-metrics --verbose --save_submission_dir submissions/$MODEL --print_submission --overwrite_submission --num_attempts=1
python cli/run_all.py --task_list_file data/task_lists/dev_list.txt --data_dir data/arc-agi/data/$SPLIT --model_configs $MODEL --log-level DEBUG --enable-metrics --submissions-root submissions/$MODEL --print_submission --num_attempts=1 --overwrite_submission
python -m src.arc_agi_benchmarking.scoring.scoring --task_dir data/arc-agi/data/$SPLIT --submission_dir submissions/$MODEL --results_dir results/$MODEL
python visualize_results.py