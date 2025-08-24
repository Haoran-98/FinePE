# -*- coding:utf-8 -*-

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
add_path = os.path.split(rootPath)[0]
sys.path.append(os.path.split(rootPath)[0])
import os
from typing import Any, Dict, List, Union, TYPE_CHECKING, Optional, Generator, Literal

from datasets import concatenate_datasets, interleave_datasets, load_dataset, load_from_disk

from log.logging import get_logger
from examples.methods.template import get_template_and_fix_tokenizer
from examples.methods.config_pair import SUPPORTED_MODELS, DEFAULT_MODULE, DEFAULT_TEMPLATE
import hashlib

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from examples.methods.hparams import ModelArguments, DataArguments
    from transformers import TrainingArguments, Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

logger = get_logger(__name__)

IGNORE_INDEX = -100

EXT2TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "txt": "text"
}


def collatefile(data_files: List[str], file_sha1: Optional[str] = None) -> None:
    if file_sha1 is None:
        logger.warning("collatefile failed: missing SHA-1 hash value in dataset_info.json.")
        return

    if len(data_files) != 1:
        logger.warning("collatefile failed: too many files.")
        return

    with open(data_files[0], "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
        if sha1 != file_sha1:
            logger.warning("collatefile failed: mismatched SHA-1 hash value at {}.".format(data_files[0]))


def split_dataset(
        dataset: Union["Dataset", "IterableDataset"],
        data_args: "DataArguments",
        training_args: "TrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6:
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset": dataset}
    else:
        return {"eval_dataset": dataset}


def get_dataset(
        model_args: "ModelArguments",
        data_args: "DataArguments"
) -> Union["Dataset", "IterableDataset"]:
    max_samples = data_args.max_samples
    all_datasets: List[Union["Dataset", "IterableDataset"]] = []

    for dataset_attr in data_args.dataset_list:
        logger.info("Loading dataset {}...".format(dataset_attr))

        if dataset_attr.load_from == "hf_hub":
            data_path = dataset_attr.dataset_name
            data_name = dataset_attr.subset
            data_files = None
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_name = dataset_attr.subset
            data_files = None
        elif dataset_attr.load_from == "file":
            data_path, data_name = None, None
            data_files: List[str] = []
            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))
                    if data_path is None:
                        data_path = EXT2TYPE.get(file_name.split(".")[-1], None)
                    else:
                        assert data_path == EXT2TYPE.get(file_name.split(".")[-1],
                                                         None), "file types are not identical."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = EXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."
            collatefile(data_files, dataset_attr.dataset_sha1)
        else:
            raise NotImplementedError

        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
            # use_auth_token=True if model_args.use_auth_token else None
        )

        if max_samples is not None:
            dataset = dataset.select(range(min(len(dataset), max_samples)))

        def convert_format(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            outputs = {"prompt": [], "query": [], "response": [], "history": []}
            for msg_list in examples[dataset_attr.messages]:
                msg_list = msg_list[:len(msg_list) // 2 * 2]
                if len(msg_list) == 0:
                    continue

                msg_pairs = []
                user_role, assistant_role = None, None
                for idx in range(0, len(msg_list), 2):
                    if user_role is None and assistant_role is None:
                        user_role = msg_list[idx][dataset_attr.role]
                        assistant_role = msg_list[idx + 1][dataset_attr.role]
                    else:
                        if (
                                msg_list[idx][dataset_attr.role] != user_role
                                or msg_list[idx + 1][dataset_attr.role] != assistant_role
                        ):
                            raise ValueError("Only accepts conversation in u/a/u/a/u/a order.")
                    msg_pairs.append((msg_list[idx][dataset_attr.content], msg_list[idx + 1][dataset_attr.content]))

                if len(msg_pairs) != 0:
                    outputs["prompt"].append(msg_pairs[-1][0])
                    outputs["query"].append("")
                    outputs["response"].append(msg_pairs[-1][1])
                    outputs["history"].append(msg_pairs[:-1])

            return outputs

        if dataset_attr.formatting == "sharegpt":
            column_names = list(next(iter(dataset)).keys())
            kwargs = {}
            if not data_args.streaming:
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=(not data_args.overwrite_cache),
                    desc="Converting format of dataset"
                )

            dataset = dataset.map(
                convert_format,
                batched=True,
                remove_columns=column_names,
                **kwargs
            )
        else:
            for column_name in ["prompt", "query", "response", "history"]:
                if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)

        if dataset_attr.system_prompt:
            system_prompt = dataset_attr.system_prompt
            if data_args.streaming:
                dataset = dataset.map(lambda _: {"system": system_prompt})
            else:
                dataset = dataset.add_column("system", [system_prompt] * len(dataset))

        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=data_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted"
        )
    else:
        raise ValueError("Unknown mixing strategy.")


def preprocess_dataset(
        dataset: Union["Dataset", "IterableDataset"],
        tokenizer: "PreTrainedTokenizer",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        model_args: "ModelArguments",
) -> Union["Dataset", "IterableDataset"]:
    template_name = DEFAULT_TEMPLATE[SUPPORTED_MODELS[model_args.model_name]]
    template = get_template_and_fix_tokenizer(template_name, tokenizer)

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                    tokenizer, query, response, history, system
            )):
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(data_args.cutoff_len * (len(source_ids) / total_len))
                max_target_len = int(data_args.cutoff_len * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_packed_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                    tokenizer, query, response, history, system
            )):
                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        total_length = len(input_ids)
        block_size = data_args.cutoff_len
        total_length = (total_length // block_size) * block_size
        for i in range(0, total_length, block_size):
            model_inputs["input_ids"].append(input_ids[i: i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i: i + block_size])

        return model_inputs

    def preprocess_unsupervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and query != ""):
                continue

            input_ids, labels = template.encode_oneturn(tokenizer, query, response, history, system)

            if template.efficient_eos:
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > data_args.cutoff_len:
                input_ids = input_ids[:data_args.cutoff_len]
            if len(labels) > data_args.cutoff_len:
                labels = labels[:data_args.cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_pairwise_dataset(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, list) and query != "" and len(response) > 1):
                continue

            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if template.efficient_eos:
                chosen_ids += [tokenizer.eos_token_id]
                rejected_ids += [tokenizer.eos_token_id]

            total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
            max_source_len = int(data_args.cutoff_len * (len(prompt_ids) / total_len))
            max_target_len = int(data_args.cutoff_len * (max(len(chosen_ids), len(rejected_ids)) / total_len))

            if len(prompt_ids) > max_source_len:
                prompt_ids = prompt_ids[:max_source_len]
            if len(chosen_ids) > max_target_len:
                chosen_ids = chosen_ids[:max_target_len]
            if len(rejected_ids) > max_target_len:
                rejected_ids = rejected_ids[:max_target_len]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)

        return model_inputs

    def print_supervised_dataset_example(example: Dict[str, List[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    def print_pairwise_dataset_example(example: Dict[str, List[int]]) -> None:
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        print("chosen_ids:\n{}".format(example["chosen_ids"]))
        print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
        print("rejected_ids:\n{}".format(example["rejected_ids"]))
        print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))

    def print_unsupervised_dataset_example(example: Dict[str, List[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if not training_args.predict_with_generate:
        preprocess_func = preprocess_packed_supervised_dataset if data_args.sft_packing else preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    else:
        preprocess_func = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    if data_args.cache_path is not None and os.path.exists(data_args.cache_path):
        logger.warning("Loading dataset from disk will ignore other data arguments.")
        return load_from_disk(data_args.cache_path)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache),
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=column_names,
            **kwargs
        )

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
            raise SystemExit("Dataset saved, rerun this script with the same `--cache_file`.")

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Empty dataset!")

        return dataset
