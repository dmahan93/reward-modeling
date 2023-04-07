import json
import torch
from datasets import Dataset, DatasetDict
import os
from huggingface_hub import login

def load_prompts(file_path):
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            loaded_line = json.loads(line)
            data.append(loaded_line["prompt"])
    return data

def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            response = json.loads(line)
            data.append(response)
    return data

def dump_jsonl(filename, data):
    with open(filename, "w") as f:
        for dict_t in data:
                json.dump(dict_t, f)
                f.write("\n")

def clean(text):
    clean_text = text.split("<|endoftext|>")[0].split("Human:")[0]
    split = clean_text.split("Assistant:")
    if len(split) > 2:
        print("Split is too long!")
        split = split[:2]
    return "Assistant:" + split[-1]


def clean_and_upload(dataset, name):
    cleaned_dataset = {"prompt": [], "response": []}
    for sample in dataset:
        # Add space since there is space at start of hh prompt
        cleaned_prompt = sample["prompt"]
        cleaned_response = "Assistant:" + sample["response"].split("<|endoftext|>")[0].split("Human:")[0].split("Assistant:")[-1]
        cleaned_dataset["prompt"].append(cleaned_prompt)
        cleaned_dataset["response"].append(cleaned_response)

    dataset = Dataset.from_dict(cleaned_dataset)
    dataset.push_to_hub(name)


def cleans(datasets, name, columns):
    cleaned_dataset = {"prompt": []}
    for column in columns:
        cleaned_dataset[column] = list()
    prompt_map = {}
    for i, dataset in enumerate(datasets):
        for sample in dataset:
            prompt_map[sample["prompt"]] = prompt_map.get(sample["prompt"], {})
            if columns[i] not in list(prompt_map[sample["prompt"]].keys()):
                prompt_map[sample["prompt"]][columns[i]] = list()
            prompt_map[sample["prompt"]][columns[i]].append(sample["response"].split("<|endoftext|>")[0].lstrip())
    prompts = list(prompt_map.keys())
    for i in range(len(prompts)):
        # Add space since there is space at start of hh prompt
        for j in range(len(prompt_map[prompts[i]][columns[0]])):
            cleaned_dataset["prompt"].append(prompts[i])
            for column in columns:
                cleaned_dataset[column].append(prompt_map[prompts[i]][column][j])

    return Dataset.from_dict(cleaned_dataset)

def clean_and_uploads(datasets, name, columns):
    cleaned_dataset = {"prompt": []}
    for column in columns:
        cleaned_dataset[column] = list()
    prompt_map = {}
    for i, dataset in enumerate(datasets):
        for sample in dataset:
            prompt_map[sample["prompt"]] = prompt_map.get(sample["prompt"], {})
            if columns[i] not in list(prompt_map[sample["prompt"]].keys()):
                prompt_map[sample["prompt"]][columns[i]] = list()
            prompt_map[sample["prompt"]][columns[i]].append(sample["response"].split("<|endoftext|>")[0].lstrip())
    prompts = list(prompt_map.keys())
    for i in range(len(prompts)):
        # Add space since there is space at start of hh prompt
        for j in range(len(prompt_map[prompts[i]][columns[0]])):
            cleaned_dataset["prompt"].append(prompts[i])
            for column in columns:
                cleaned_dataset[column].append(prompt_map[prompts[i]][column][j])

    dataset = Dataset.from_dict(cleaned_dataset)
    print(name)
    dataset.push_to_hub(name)


def clean_and_upload_train_test_valid2(trains, tests, valids1, valids2, name, columns):
    cleaned_dataset1 = {"prompt": []}
    cleaned_dataset2 = {"prompt": []}
    for column in columns:
        cleaned_dataset2[column] = list()
        cleaned_dataset1[column] = list()
    prompt_map = {}
    for i, dataset in enumerate(trains):
        for sample in dataset:
            prompt_map[sample["prompt"]] = prompt_map.get(sample["prompt"], {})
            if columns[i] not in list(prompt_map[sample["prompt"]].keys()):
                prompt_map[sample["prompt"]][columns[i]] = list()
            prompt_map[sample["prompt"]][columns[i]].append(sample["response"].split("<|endoftext|>")[0].lstrip())
    prompts = list(prompt_map.keys())
    for i in range(len(prompts)//2):
        # Add space since there is space at start of hh prompt
        for j in range(len(prompt_map[prompts[i]][columns[0]])):
            cleaned_dataset1["prompt"].append(prompts[i])
            for column in columns:
                cleaned_dataset1[column].append(prompt_map[prompts[i]][column][j])
    for i in range(len(prompts)//2, len(prompts)):
        # Add space since there is space at start of hh prompt
        for j in range(len(prompt_map[prompts[i]][columns[0]])):
            cleaned_dataset2["prompt"].append(prompts[i])
            for column in columns:
                cleaned_dataset2[column].append(prompt_map[prompts[i]][column][j])

    dataset1 = DatasetDict({'train': Dataset.from_dict(cleaned_dataset1, split='train'),
                          'test': cleans(tests, name, columns),
                          'valid': cleans(valids1, name, columns)}).push_to_hub(f"{name}-split1")
    dataset2 = DatasetDict({'train': Dataset.from_dict(cleaned_dataset2, split='train'),
                          'test': cleans(tests, name, columns),
                          'valid': cleans(valids2, name, columns)}).push_to_hub(f"{name}-split2")


def clean_rm_and_upload(dataset, name):
    cleaned_dataset = {"response": [], "reward": []}
    for sample in dataset:
        response = sample["prompt"].replace("<|endoftext|>", "")
        cleaned_dataset["response"].append(response)
        cleaned_dataset["reward"].append(sample["reward"])
    dataset = Dataset.from_dict(cleaned_dataset)
    dataset.push_to_hub(name)


if __name__ == "__main__":
    '''files = os.listdir("datasets")
    for file in files:
        dataset = load_jsonl(os.path.join("datasets", file))
        name = file.replace(".jsonl", "")
        print("Processing {}...".format(name))
        clean_and_upload(dataset, name)'''
    tests = [
        load_jsonl(r"..\pythia_125M_sft_summarize_comparisons_test.jsonl"),
        load_jsonl(r"..\pythia_1B_sft_summarize_comparisons_test.jsonl"),
        load_jsonl(r"..\pythia_6B_sft_summarize_comparisons_test.jsonl"),
        load_jsonl(r"..\pythia_20B_sft_summarize_comparisons_test.jsonl")
    ]
    valid1s = [
        load_jsonl(r"..\pythia_125M_sft_summarize_comparisons_valid1.jsonl"),
        load_jsonl(r"..\pythia_1B_sft_summarize_comparisons_valid1.jsonl"),
        load_jsonl(r"..\pythia_6B_sft_summarize_comparisons_valid1.jsonl"),
        load_jsonl(r"..\pythia_20B_sft_summarize_comparisons_valid1.jsonl")
    ]
    valid2s = [
        load_jsonl(r"..\pythia_125M_sft_summarize_comparisons_valid2.jsonl"),
        load_jsonl(r"..\pythia_1B_sft_summarize_comparisons_valid2.jsonl"),
        load_jsonl(r"..\pythia_6B_sft_summarize_comparisons_valid2.jsonl"),
        load_jsonl(r"..\pythia_20B_sft_summarize_comparisons_valid2.jsonl")
    ]
    trains = [
        load_jsonl(r"..\pythia_125M_sft_summarize_comparisons_train.jsonl"),
        load_jsonl(r"..\pythia_1B_sft_summarize_comparisons_train.jsonl"),
        load_jsonl(r"..\pythia_6B_sft_summarize_comparisons_train.jsonl"),
        load_jsonl(r"..\pythia_20B_sft_summarize_comparisons_train.jsonl")
    ]
    clean_and_upload_train_test_valid2(trains, tests, valid1s, valid2s, "dmayhem93/summarization-sft-heirarchical", ["125M", "1B", "6B", "20B"])
    # clean_rm_and_upload(dataset, "dmayhem93/summarization_125M")