# -*- coding:utf-8 -*-
import sys
import os
import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, AutoConfig, AutoModelForCausalLM, \
    AutoTokenizer

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.dataset import get_dataset, preprocess_dataset
from utils.moe_lora import add_moelora_to_model, moeLoRAConfig

from log.logging import get_logger

from utils.tool import accumulative_multiplication
if TYPE_CHECKING:
    from examples.methods.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments

IGNORE_INDEX = -100
logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_sft(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
):
    dataset = get_dataset(model_args, data_args)
    # model_created, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args, data_args)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        trust_remote_code=False,
        use_flash_attention_2=False,
        torch_dtype=torch.bfloat16,
    ).to(device)
    config = AutoConfig.from_pretrained(
        model_args.model_path,
        trust_remote_code=False,
        use_flash_attention_2=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, model_args)
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, drop_last=True,
                            collate_fn=lambda batch: data_collator(batch, 'pt'))
    
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    
    adapter_checkpoint_dict = {}
    
    checkpoint_dir_path = finetuning_args.adapter_dir
    checkpoint_names = os.listdir(checkpoint_dir_path)
    for index, checkpoint_name in enumerate(checkpoint_names):
        checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
        if checkpoint_name not in adapter_checkpoint_dict:
            adapter_checkpoint_dict[f'adapter_{index}'] = checkpoint_path
    
    logger.info(f"adapter_checkpoint_dict:{adapter_checkpoint_dict}")
    # Convert the model to X-LoRA
    model_created = add_moelora_to_model(
        model=model,
        moelora_config=moeLoRAConfig(
            config.hidden_size,
            base_model_id=model_args.model_name,
            device=device,
            adapters=adapter_checkpoint_dict,
        ),
        verbose=True,
    )
    
    # print_model_parameters(model_created)
    
    print(model_created)
    
    # Use trainable adapters: mark all adapters as trainable
    model_created.set_use_trainable_adapters(False)
    
    # print_model_parameters(model_created)
    
    # Set the scaling pass value to 0, meaning that no adapters will contribute to the scaling pass output
    model_created.set_scaling_pass_value(None)
    
    # Multiply the output of each LoRA adapter by 2, additionally to the scalings.
    model_created.set_global_scaling_weight(1)
    
    # Returns 2
    logger.info(f"global_scaling_weight:{model_created.get_global_scaling_weight()}")
    
    model_created.print_trainable_parameters()
    
    # Enable scalings logging and begin a log
    model_created.enable_scalings_logging()
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model_created.parameters(), training_args.learning_rate)
    
    model_created.train()
    
    total_loss = 0
    total_step = 0
    
    time = datetime.datetime.now()
    logging_steps = 5  # 每隔多少步输出
    max_grad_norm = 1.0  # 最大梯度范数，用于剪裁
    
    log_dir_for_autodl = "/root/tf-logs/"
    log_dir = f'../log/tensorboard_log/'
    # 创建一个 SummaryWriter 实例
    writer = SummaryWriter(log_dir=log_dir_for_autodl + time.strftime(
        "%m_%d_%H_%M") + f'_{data_args.dataset}_combination' + '/')  # save log to this dir
    for epoch in range(training_args.num_train_epochs):
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{training_args.num_train_epochs}")):
            input_ids = batch['input_ids'][:-1].to(device)
            attention_mask = batch['attention_mask'][:-1].to(device)
            labels = batch['labels'][:-1].to(device)
            
            optimizer.zero_grad()
            # 看似调用模型前馈，其实此处仅仅使用的是基础模型的前馈，
            outputs = model_created(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logger.info(f"loss: {loss.item()}")
            
            #  Calculate the value of personality balance loss
            moelora_scalings = model_created.internal_moelora_scalings.mean(dim=[0, 1, 2])
            
            # split moelora_scalings to 6 lists
            # moelora_scalings_split_lists = split_list(moelora_scalings)
            # print("moelora_scalings_split_lists=", moelora_scalings_split_lists)
            personality_balance_loss = 0.0
            # for i, sublist in enumerate(moelora_scalings_split_lists):
            personality_balance_loss += accumulative_multiplication(moelora_scalings)
            logger.info(f"personality_balance_loss: {personality_balance_loss}")
            
            total_loss = total_loss + loss.item()
            loss = loss + moeLoRAConfig.loss_balance_scale * personality_balance_loss
            logger.info(f"Overall_loss: {loss}")
            
            loss.backward()
            
            # 计算梯度范数并裁剪
            total_norm = nn.utils.clip_grad_norm_(model_created.parameters(), max_grad_norm)
            torch.autograd.set_detect_anomaly(True)
            optimizer.step()
            
            total_step = total_step + 1
            # # Run forward passes to accumulate a log
            #
            # # Write the log to a file, or multiple.
            model_created.flush_log_scalings(
                f"../log/save_log/" + time.strftime("%Y-%m-%d-%H-%M-%S") + f"_{data_args.dataset}_scalings")
            avg_loss = total_loss / total_step
            if total_step % logging_steps == 0:
                print(f"Step: {step + 1}/{len(dataset)}, Loss: {avg_loss:.4f}, Grad Norm: {total_norm:.4f}",
                      flush=True)
            writer.add_scalar('Avg_loss/train', avg_loss, total_step)
            
            #  To display formatted tensors on TensorBoard
            tag_scalar_dict = {}
            # tag_scalar_dict = {'adapter_0': 0.1, 'adapter_1': 0.2, ...}
            
            for index, (key, value) in enumerate(adapter_checkpoint_dict.items()):
                tag_scalar_dict[key] = moelora_scalings[index]
            logger.info(f'tag_scalar_dict: \n{tag_scalar_dict}')
            
            writer.add_scalars('Scaling_adapters', tag_scalar_dict, global_step=total_step)
            writer.add_histogram('Scaling/train', model_created.internal_moelora_scalings, total_step)
            # 将张量添加到 TensorBoard
            writer.add_scalar('moelora_scalings_mean', model_created.internal_moelora_scalings.mean(), total_step)
            writer.add_scalar('moelora_scalings_std', model_created.internal_moelora_scalings.std(), total_step)
            
            # 某一权重超过0.9则保存门控权重不再训练
            for key, value in tag_scalar_dict.items():
                weight = value.float()
                # print(f'{key}: {weight:.4f}'"")
                if weight > 0.85:
                    model_created.save_pretrained(
                        f'saves/{time.strftime("%Y-%m-%d-%H-%M-%S")}_{data_args.dataset}_pertrain')
                    return
                    
        # 打印每个 epoch 结束的累计损失
        print(f"Epoch {epoch + 1} finished, Average Loss: {total_loss / total_step:.4f}", flush=True)
        # 每一个 epoch 保存一个 classifier 权重和配置
        model_created.save_pretrained(
            f'saves/{time.strftime("%Y-%m-%d-%H-%M-%S")}_{data_args.dataset}_pertrain')
    
    # 关闭 writer
    writer.close()


from examples.methods.parser import get_train_args


def main():
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)


if __name__ == "__main__":
    main()
