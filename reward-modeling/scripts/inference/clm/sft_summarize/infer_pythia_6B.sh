accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_6B_sft_summarize_comparisons_train \
--model_name dmayhem93/pythia-6B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split train \
--batch_size 2
accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_6B_sft_summarize_comparisons_valid1 \
--model_name dmayhem93/pythia-6B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split valid1 \
--batch_size 2
accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_6B_sft_summarize_comparisons_valid2 \
--model_name dmayhem93/pythia-6B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split valid2 \
--batch_size 2
accelerate launch --config_file scripts/inference/accelerate_configs/default.yaml \
inference.py \
--prompt_dataset CarperAI/openai_summarize_comparisons \
--log_file pythia_6B_sft_summarize_comparisons_test \
--model_name dmayhem93/pythia-6B-Summarization-sft \
--tokenizer_name EleutherAI/gpt-neox-20b \
--split test \
--batch_size 2