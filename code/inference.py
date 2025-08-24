# -*- coding:utf-8 -*-
import argparse
import json
import sys
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, AutoConfig, AutoModelForCausalLM, \
    AutoTokenizer

from utils.tool import find_numbers_in_first_five_chars, get_content_after_last_slash_mark

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.moe_lora import moeLoRAConfig, from_pretrained, use_classifier_obtain_weight
from log.logging import get_logger

if TYPE_CHECKING:
    from examples.methods.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

IGNORE_INDEX = -100
logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_infer(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        generating_args: "GeneratingArguments",
):
    # dataset = get_dataset(model_args, data_args)
    training_args.predict_with_generate = True
    model = AutoModelForCausalLM.from_pretrained(
        "../Meta-Llama-3-8B-Instruct",
        trust_remote_code=False,
        use_flash_attention_2=False,
        torch_dtype=torch.bfloat16,
    ).to(device)
    config = AutoConfig.from_pretrained(
        "../Meta-Llama-3-8B-Instruct",
        trust_remote_code=False,
        use_flash_attention_2=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("../Meta-Llama-3-8B-Instruct")
    # dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, model_args)
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
    
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=tokenizer,
    #     pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
    #     label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # )
    # dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True,
    #                         collate_fn=lambda batch: data_collator(batch, 'pt'))
    
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    
    adapter_checkpoint_dict = {}
    
    checkpoint_dir_path = generating_args.infer_adapter_dir
    # checkpoint_dir_path = infer_adapter_dir
    checkpoint_names = os.listdir(checkpoint_dir_path)
    for index, checkpoint_name in enumerate(checkpoint_names):
        checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
        if checkpoint_name not in adapter_checkpoint_dict:
            adapter_checkpoint_dict[f'adapter_{index}'] = checkpoint_path
    
    logger.info(f"adapter_checkpoint_dict:{adapter_checkpoint_dict}")
    # Convert the model to X-LoRA
    model_peft, classifier = from_pretrained(
        model=model,
        moelora_config=moeLoRAConfig(
            config.hidden_size,
            base_model_id=model_args.model_name,
            device=device,
            adapters=adapter_checkpoint_dict,
            # classifier_path=classifier_path,
            # global_scaling_weight=global_scaling_weight
        ),
        device=device,
        verbose=True,
    )
    
    # 将模型设置为评估模式
    classifier.eval()
    model_peft.eval()
    
    with torch.no_grad():  # 确保不会计算梯度
        # MPI and BFI Simultaneous execution
        with open(os.path.join(data_args.dataset_dir, "BFI_ds.json"), 'r',
                  encoding='utf-8') as dataset_json_file:
            dataset_json = json.load(dataset_json_file)
        # IFeval
        # dataset_json = []
        # with open(os.path.join(data_args.dataset_dir, "BBQ.jsonl"), 'r',
        #           encoding='utf-8') as dataset_jsonl_file:
        #     for line in dataset_jsonl_file:
        #         print(line)
        #         line_json = json.loads(line)
        #         dataset_json.append(line_json)
        # 使用模型生成文本
        result_dict = {"response": {}}
        with open(f"../data/eval_ds_raw/BFI.json", 'r',
                  encoding='utf-8') as f:
            data_psychobench = json.load(f)
        categories = data_psychobench['categories']
        
        for category in categories:
            cat_name = category['cat_name']
            cat_questions = category['cat_questions']
            if generating_args.infer_adapter_dir.split('/')[-1] == cat_name:
                adj_list = cat_questions
        
        for epoch in range(1):
            result_dict["response"][epoch] = []
            for index, value in enumerate(tqdm(dataset_json, desc=f"Epoch {epoch + 1}/{10}")):
                
                if index + 1 not in adj_list:
                    continue
                # 确保pad_token_id被设置
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                system_message = ("Imagine you are a real person, not just a language model. Now, you’re asked the "
                                  "following question. Respond based on your authentic thoughts and emotions—don’t "
                                  "overthink it.")
                user_message = f"{value['input_rewritten']}"
                system_message = system_message + (
                    f"Reply a number from 1 to 5 using the scales: 1 denotes 'strongly disagree', "
                    f"2 denotes 'a little disagree', 3 denotes 'neither agree nor disagree', "
                    f"4 denotes 'little agree', and 5 denotes 'strongly agree'. "
                    f"Give the answer first and then provide an explanation.")
                format_messages = (f"<begin_of_text><|start_header_id|>system<|end_header_id|>\n{system_message}"
                                   f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n  {user_message} <|eot_id|>"
                                   f"<|start_header_id|>assistant<|end_header_id|>\n")
                logger.info(f"messages: \n{format_messages}\n")
                encoded_input = tokenizer([format_messages], return_tensors='pt', padding=True, truncation=True).to(
                    'cuda')
                
                # 初始化生成的序列
                generated = encoded_input['input_ids']
                
                inputs = generated[:, -data_args.cutoff_len:]  # 保证输入序列长度在模型限制内
                
                # next_token = ""
                result_list = []
                mean_scalings_dict = {"mean_scalings_list": []}
                
                repetition_penalty = 1.5
                no_repeat_ngram_size = 3
                ngram_history = defaultdict(set)  # 存储历史 n-gram
                
                # 开始手动生成过程
                for _ in range(20):
                    # 初始化分类器权重
                    use_classifier_obtain_weight(model_peft, classifier, inputs, mean_scalings_dict)
                    
                    # 获取当前token的logits，预测下一个token
                    with torch.no_grad():  # 禁用梯度计算
                        logits = model_peft(inputs).logits[:, -1, :]  # 获取最后一个token的logits
                    
                    # 设置温度参数（通常大于0，小于1会增加随机性，大于1会减少随机性）
                    temperature = 0.01  # 可以根据需要调整这个值
                    
                    # 应用温度参数
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # 转换为概率分布并采样
                    probabilities = torch.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probabilities, num_samples=1)
                    
                    # # # 获取下一个token的ID（贪婪解码：选择最大概率的token）
                    # next_token_id = torch.argmax(logits, dim=-1)
                    
                    # --- 重复惩罚 ---
                    for i, token_id in enumerate(generated[0]):
                        logits[0, token_id] /= repetition_penalty
                    
                    # --- no_repeat_ngram 限制 ---
                    if generated.size(1) >= no_repeat_ngram_size - 1:
                        current_prefix = tuple(generated[0, -no_repeat_ngram_size + 1:].tolist())
                        banned_tokens = ngram_history.get(current_prefix, set())
                        logits[0, list(banned_tokens)] = float('-inf')
                    
                    # 解码生成的token为文本
                    next_token = tokenizer.decode(next_token_id[0])
                    
                    # 打印生成的token和它的ID
                    logger.info(f"Generated token: {next_token} (ID: {next_token_id.item()})")
                    
                    eos_token_id = tokenizer.eos_token_id or 128009
                    if next_token_id.item() in {128009, eos_token_id}:
                        break
                    # 将生成的token ID加入到当前生成的序列
                    # generated = torch.cat((generated, next_token_id.unsqueeze(-1)), dim=-1)
                    # use 温度控制
                    generated = torch.cat((generated, next_token_id), dim=-1)
                    
                    # 更新 ngram history
                    if generated.size(1) >= no_repeat_ngram_size:
                        ngram = tuple(generated[0, -no_repeat_ngram_size:].tolist())
                        prefix = ngram[:-1]
                        ngram_history[prefix].add(ngram[-1])
                    
                    inputs = generated[:, -data_args.cutoff_len:]  # 保证输入序列长度在模型限制内
                    
                    result_list.append(next_token)
                
                # 最终生成的文本
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                logger.info(f"\nFinal generated text:{generated_text}")
                result = ''.join(result_list)
                logger.info(f"result:{result}")
                if find_numbers_in_first_five_chars(result) is not None:
                    logger.info(f"------------------result-------------: {result}")
                    result_dict["response"][epoch].append(find_numbers_in_first_five_chars(result))
                else:
                    result_dict["response"][epoch].append(0)
                # result_dict["response"][epoch].append(user_message + " Response: " + result)
                logger.info(f"result_dict: {result_dict}")
                # # for IFeval
                # value["response"] = result
                
                # total_list = [0] * len(mean_scalings_dict["mean_scalings_list"][0])
                # for mean_scalings in mean_scalings_dict["mean_scalings_list"]:
                #     total_list = [a + b for a, b in zip(total_list, mean_scalings)]
                # total_mean_scaling = [x / len(mean_scalings_dict["mean_scalings_list"]) for x in total_list]
                # mean_scalings_dict['total_mean_scaling'] = total_mean_scaling
                # logger.info(f"------------------total_mean_scaling-------------: {total_mean_scaling}")
                # with open(
                #         f"eval_results/scalings/mean_scalings_{get_content_after_last_slash_mark(generating_args.infer_adapter_dir)}.json",
                #         'w', encoding='utf-8') as f:
                #     json.dump(mean_scalings_dict, f, indent=4, ensure_ascii=False)
                # break
            
            # 生成文本
            # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # generated_text = generated_text[
            #                  len(tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)):]
            # logger.info(f"generated_text: {generated_text}")
            # logger.info("\n")
            # # Check the format of generated_text, and output an int number as the standard
            logger.info(f"result_dict: {result_dict}")
            
            # for IFeval
            # value["response"] = result
            # with open(
            #         f"eval_results/BBQ/BBQ_result_{get_content_after_last_slash_mark(infer_adapter_dir)}_{global_scaling_weight}.jsonl",
            #         'a', encoding="utf-8") as f:
            #     for item in dataset_json:
            #         f.write(json.dumps(item))
            #         f.write("\n")
            result_dict["global_scaling_weight"] = classifier.config.global_scaling_weight
            result_dict["prompt"] = format_messages
            with open(
                    f"eval_results/combination/eval_result_BFI_ds_{get_content_after_last_slash_mark(generating_args.infer_adapter_dir)}.json",
                    "w",
                    encoding="utf-8") as eval_result_file:
                json.dump(result_dict, eval_result_file, ensure_ascii=False, indent=4)


def run_infer_open_task(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        infer_adapter_dir: str,
        classifier_path: str,
        global_scaling_weight: float
):
    # dataset = get_dataset(model_args, data_args)
    training_args.predict_with_generate = True
    model = AutoModelForCausalLM.from_pretrained(
        "../Meta-Llama-3-8B-Instruct",
        trust_remote_code=False,
        use_flash_attention_2=False,
        torch_dtype=torch.bfloat16,
    ).to(device)
    config = AutoConfig.from_pretrained(
        "../Meta-Llama-3-8B-Instruct",
        trust_remote_code=False,
        use_flash_attention_2=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("../Meta-Llama-3-8B-Instruct")
    # dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, model_args)
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
    
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=tokenizer,
    #     pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
    #     label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # )
    # dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True,
    #                         collate_fn=lambda batch: data_collator(batch, 'pt'))
    
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    
    adapter_checkpoint_dict = {}
    
    # checkpoint_dir_path = generating_args.infer_adapter_dir
    checkpoint_dir_path = infer_adapter_dir
    checkpoint_names = os.listdir(checkpoint_dir_path)
    for index, checkpoint_name in enumerate(checkpoint_names):
        checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
        if checkpoint_name not in adapter_checkpoint_dict:
            adapter_checkpoint_dict[f'adapter_{index}'] = checkpoint_path
    
    logger.info(f"adapter_checkpoint_dict:{adapter_checkpoint_dict}")
    # Convert the model to X-LoRA
    model_peft, classifier = from_pretrained(
        model=model,
        moelora_config=moeLoRAConfig(
            config.hidden_size,
            base_model_id=model_args.model_name,
            device=device,
            adapters=adapter_checkpoint_dict,
            classifier_path=classifier_path,
            global_scaling_weight=global_scaling_weight
        ),
        device=device,
        verbose=True,
    )
    
    # 将模型设置为评估模式
    classifier.eval()
    model_peft.eval()

    with torch.no_grad():  # 确保不会计算梯度
        # MPI and BFI Simultaneous execution
        # with open(os.path.join(data_args.dataset_dir, "BFI_ds.json"), 'r',
        #           encoding='utf-8') as dataset_json_file:
        #     dataset_json = json.load(dataset_json_file)
        # IFeval
        dataset_json = []
        with open(os.path.join(data_args.dataset_dir, "open_task_writing/case_list.jsonl"), 'r',
                  encoding='utf-8') as dataset_jsonl_file:
            for line in dataset_jsonl_file:
                print(line)
                line_json = json.loads(line)
                dataset_json.append(line_json)
        dataset_json = dataset_json[:20]
        # 使用模型生成文本
        result_dict = {"response": {}}
        # with open(f"../data/eval_ds_raw/BFI.json", 'r',
        #           encoding='utf-8') as f:
        #     data_psychobench = json.load(f)
        # categories = data_psychobench['categories']
        # adj_list = [1]
        # for category in categories:
        #     cat_name = category['cat_name']
        #     cat_questions = category['cat_questions']
        #     if generating_args.infer_adapter_dir.split('/')[-1] == cat_name:
        #         adj_list = cat_questions
        
        for epoch in range(1):
            result_dict["response"][epoch] = []
            for index, value in enumerate(tqdm(dataset_json, desc=f"Epoch {epoch + 1}/{10}")):
                
                # if index + 1 not in adj_list:
                #     continue
                # 确保pad_token_id被设置
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                system_message = "Now, you’re asked the following question. Respond based on your authentic thoughts and emotions—don’t overthink it. \nLet your words flow naturally, focusing on expressing your genuine feelings and reactions. Aim to keep your response under 256 words."
                user_message = (f"{value['user']}")
                format_messages = (f"<begin_of_text><|start_header_id|>system<|end_header_id|>\n{system_message}"
                                   f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n  {user_message} <|eot_id|>"
                                   f"<|start_header_id|>assistant<|end_header_id|>\n")
                logger.info(f"messages: \n{format_messages}\n")
                encoded_input = tokenizer([format_messages], return_tensors='pt', padding=True, truncation=True).to(
                    'cuda')
                
                # 初始化生成的序列
                generated = encoded_input['input_ids']
                
                inputs = generated[:, -data_args.cutoff_len:]  # 保证输入序列长度在模型限制内
                
                # next_token = ""
                result_list = []
                mean_scalings_dict = {"mean_scalings_list": []}
                
                repetition_penalty = 1.5
                no_repeat_ngram_size = 3
                ngram_history = defaultdict(set)  # 存储历史 n-gram
                
                # 开始手动生成过程
                for _ in range(256):
                    # 初始化分类器权重
                    use_classifier_obtain_weight(model_peft, classifier, inputs, mean_scalings_dict)
                    
                    # 获取当前token的logits，预测下一个token
                    with torch.no_grad():  # 禁用梯度计算
                        logits = model_peft(inputs).logits[:, -1, :]  # 获取最后一个token的logits
                    
                    # 设置温度参数（通常大于0，小于1会增加随机性，大于1会减少随机性）
                    temperature = 0.6  # 可以根据需要调整这个值
                    
                    # 应用温度参数
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # 转换为概率分布并采样
                    probabilities = torch.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probabilities, num_samples=1)
                    
                    # # # 获取下一个token的ID（贪婪解码：选择最大概率的token）
                    # next_token_id = torch.argmax(logits, dim=-1)
                    
                    # --- 重复惩罚 ---
                    for i, token_id in enumerate(generated[0]):
                        logits[0, token_id] /= repetition_penalty
                    
                    # --- no_repeat_ngram 限制 ---
                    if generated.size(1) >= no_repeat_ngram_size - 1:
                        current_prefix = tuple(generated[0, -no_repeat_ngram_size + 1:].tolist())
                        banned_tokens = ngram_history.get(current_prefix, set())
                        logits[0, list(banned_tokens)] = float('-inf')
                    
                    # 解码生成的token为文本
                    next_token = tokenizer.decode(next_token_id[0])
                    
                    # 打印生成的token和它的ID
                    logger.info(f"Generated token: {next_token} (ID: {next_token_id.item()})")
                    
                    eos_token_id = tokenizer.eos_token_id or 128009
                    if next_token_id.item() in {128009, eos_token_id}:
                        break
                    # 将生成的token ID加入到当前生成的序列
                    # generated = torch.cat((generated, next_token_id.unsqueeze(-1)), dim=-1)
                    # use 温度控制
                    generated = torch.cat((generated, next_token_id), dim=-1)
                    
                    # 更新 ngram history
                    if generated.size(1) >= no_repeat_ngram_size:
                        ngram = tuple(generated[0, -no_repeat_ngram_size:].tolist())
                        prefix = ngram[:-1]
                        ngram_history[prefix].add(ngram[-1])
                    
                    inputs = generated[:, -data_args.cutoff_len:]  # 保证输入序列长度在模型限制内
                    
                    result_list.append(next_token)
                
                # 最终生成的文本
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                logger.info(f"\nFinal generated text:{generated_text}")
                result = ''.join(result_list)
                logger.info(f"result:{result}")
                # if find_numbers_in_first_five_chars(result) is not None:
                #     logger.info(f"------------------result-------------: {result}")
                #     result_dict["response"][epoch].append(find_numbers_in_first_five_chars(result))
                # else:
                #     result_dict["response"][epoch].append(0)
                result_dict["response"][epoch].append(user_message + " Response: " + result)
                logger.info(f"result_dict: {result_dict}")
                # # for IFeval
                value["response"] = result
                
                # total_list = [0] * len(mean_scalings_dict["mean_scalings_list"][0])
                # for mean_scalings in mean_scalings_dict["mean_scalings_list"]:
                #     total_list = [a + b for a, b in zip(total_list, mean_scalings)]
                # total_mean_scaling = [x / len(mean_scalings_dict["mean_scalings_list"]) for x in total_list]
                # mean_scalings_dict['total_mean_scaling'] = total_mean_scaling
                # logger.info(f"------------------total_mean_scaling-------------: {total_mean_scaling}")
                # with open(
                #         f"eval_results/scalings/mean_scalings_{get_content_after_last_slash_mark(generating_args.infer_adapter_dir)}.json",
                #         'w', encoding='utf-8') as f:
                #     json.dump(mean_scalings_dict, f, indent=4, ensure_ascii=False)
                # break
        
        # 生成文本
        # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # generated_text = generated_text[
        #                  len(tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)):]
        # logger.info(f"generated_text: {generated_text}")
        # logger.info("\n")
        # # Check the format of generated_text, and output an int number as the standard
        # logger.info(f"result_dict: {result_dict}")
        
        # for IFeval
        # value["response"] = result
        with open(
                f"eval_results/open_task/{get_content_after_last_slash_mark(generating_args.infer_adapter_dir)}-.jsonl",
                'a', encoding="utf-8") as f:
            for item in dataset_json:
                f.write(json.dumps(item))
                f.write("\n")
        result_dict["global_scaling_weight"] = classifier.config.global_scaling_weight
        result_dict["prompt"] = format_messages
        with open(
                f"eval_results/open_task/eval_result_BFI_ds_{get_content_after_last_slash_mark(generating_args.infer_adapter_dir)}-.json",
                "w",
                encoding="utf-8") as eval_result_file:
            json.dump(result_dict, eval_result_file, ensure_ascii=False, indent=4)


from examples.methods.parser import get_train_args

if __name__ == "__main__":
    trait_list = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    global_scaling_weight_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    
    for index, trait in enumerate(trait_list):
        run_infer_open_task(model_args, data_args, training_args, finetuning_args, generating_args,
                            f"",
                            f"",
                            global_scaling_weight_list[index])
    run_infer(model_args, data_args, training_args, finetuning_args, generating_args)
