# Source: https://www.datacamp.com/tutorial/fine-tuning-llama-2

import os
import yaml
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import AutoPeftModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer
from random import randrange

torch.cuda.empty_cache()

def parse_command_args():
    """Parse command-line arguments """
    parser = argparse.ArgumentParser(description='Model fine-tuning')
    parser.add_argument('-m', '--model', required=True,
                        help='model to finetune', 
                        choices=['llama7', 'llama13', 'mistral', 'mixtral', 'phi2'])
    parser.add_argument('-d', '--dataset', required=True,
                        help='path to dataset')

    args = parser.parse_args()
    return args


def load_training_params(config, args):
    """Load training parameters from config """
    training_params = TrainingArguments(
        output_dir = os.path.join(config['train']['out_dir'], args.model, args.dataset,),
        num_train_epochs = config['train']['epochs'],
        per_device_train_batch_size = config['train']['batch_size'],
        gradient_accumulation_steps = config['train']['grad_acc_steps'],
        optim = "paged_adamw_32bit",
        save_steps = config['train']['save_steps'],
        logging_steps = config['train']['log_steps'],
        learning_rate = config['train']['lr'],
        weight_decay = config['train']['decay'],
        fp16 = False,
        bf16 = False,
        max_grad_norm = 0.3,
        max_steps = -1,
        warmup_ratio = 0.03,
        group_by_length = True,
        lr_scheduler_type = "constant",
    )
    return training_params


def load_quant_config():
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False, # Changed
    )
    return quant_config


def load_peft_params():
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft_params


def load_model(base_model):
    quant_config = load_quant_config()

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def format_instruction(sample):
    formatted_string = f"""Instruction: {sample['instruction']}
              Input: {sample['input']}
              Response: {sample['response']}
"""
    return [formatted_string]  # Return as a list


def finetune(config, args):
    dataset = load_dataset("csv", data_files=config['dataset'][args.dataset], split='train')

    base_model = config['model'][args.model]
    model, tokenizer = load_model(base_model)
    peft_params = load_peft_params()
    training_params = load_training_params(config, args)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        # dataset_text_field="text",
        formatting_func=format_instruction,
        max_seq_length= config['train']['max_seq_len'],
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    trainer.train()
    print("Training Complete")
    trainer.save_model()
    print("Model Saved")
    return


def main():
    args = parse_command_args()
    
    # Load yaml file
    with open('biased_flawed/instr_tuning/config_finetune.yaml') as file:
        config = yaml.safe_load(file)

    base_model = config['model'][args.model]
    data = config['dataset'][args.dataset]

    finetune(config, args)

    # grid_search_test(config, args)

    return


if __name__ == '__main__':
    main()
