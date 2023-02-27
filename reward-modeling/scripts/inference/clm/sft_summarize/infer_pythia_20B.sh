accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_20B_sft_summarize_comparisons_train \
--model_name dmayhem93/neox-20B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 2
accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_20B_sft_summarize_comparisons_valid1 \
--model_name dmayhem93/neox-20B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split valid1 \
--batch_size 2
accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_20B_sft_summarize_comparisons_valid2 \
--model_name dmayhem93/neox-20B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split valid2 \
--batch_size 2
accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_20B_sft_summarize_comparisons_test \
--model_name dmayhem93/neox-20B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split test \
--batch_size 2
