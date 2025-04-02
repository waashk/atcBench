
import torch
import random

from peft import LoraConfig
from transformers import BitsAndBytesConfig

import evaluate
import numpy as np

# Activate 4-bit precision base model loading
def create_bnb_config(load_in_4bit = True, 
                      bnb_4bit_use_double_quant = True, 
                      bnb_4bit_quant_type = "nf4", 
                      bnb_4bit_compute_dtype = torch.bfloat16):
    """
    Configures model quantization method using bitsandbytes to speed up training and inference

    :param load_in_4bit: Load model in 4-bit precision mode
    :param bnb_4bit_use_double_quant: Nested quantization for 4-bit model
    :param bnb_4bit_quant_type: Quantization data type for 4-bit model
    :param bnb_4bit_compute_dtype: Computation data type for 4-bit model
    """
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

    return bnb_config








def create_peft_config(r, lora_alpha, lora_dropout, bias, task_type, target_modules=None, layers_to_transform=None):
    """
    Creates Parameter-Efficient Fine-Tuning configuration for the model

    :param r: LoRA attention dimension
    :param lora_alpha: Alpha parameter for LoRA scaling
    :param modules: Names of the modules to apply LoRA to
    :param lora_dropout: Dropout Probability for LoRA layers
    :param bias: Specifies if the bias parameters should be trained
    """
    if target_modules:
        config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
        target_modules=target_modules, 
        layers_to_transform=layers_to_transform
        )
    else:
        config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = task_type,
            #target_modules="all-linear", 
        )

    return config





#preprocess_batch
def tokenize_function(dataset, tokenizer, max_length):
#def tokenize_function(dataset):
    """
    Tokenizes dataset 

    :param dataset: Dataset 
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        dataset["text"],
        max_length = max_length,
        truncation = True,
        padding = "max_length"
    )


metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro') 