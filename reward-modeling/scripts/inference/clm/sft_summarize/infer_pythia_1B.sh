accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_1B_sft_summarize_eval \
--model_name dmayhem93/pythia-1B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 2
