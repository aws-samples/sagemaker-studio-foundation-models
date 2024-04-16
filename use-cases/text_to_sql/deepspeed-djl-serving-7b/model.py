import copy
import sqlparse
import boto3
from djl_python import Input, Output
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, Tuple
import deepspeed
import warnings
import tarfile

predictor = None
prompt_for_db_dict_cache = {}

def download_prompt_from_s3(prompt_filename):

    print(f"downloading prompt file: {prompt_filename}")
    s3 = boto3.resource('s3')
    
    obj = s3.Object("sagemaker-us-east-2-968192116650", f"database-prompts/{prompt_filename}")
    file_content = obj.get()['Body'].read().decode('utf-8')

    print(f"downloaded prompt file: {prompt_filename}!")

    return file_content


def get_model(properties):
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    cwd = properties["model_id"]

    print(cwd, os.listdir(cwd))
    
    print(f"Loading model from {cwd}")
    model = AutoModelForCausalLM.from_pretrained(
        cwd, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16
    )
    model = deepspeed.init_inference(
        model, 
        mp_size=properties["tensor_parallel_degree"]
    )
    
    print(f"Loading tokenizer from {cwd}")
    tokenizer = AutoTokenizer.from_pretrained(cwd)
    generator = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=local_rank,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return generator


def handle(inputs: Input) -> None:
    global predictor
    if not predictor:
        predictor = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_json()
    
    text = data["inputs"]
    generation_kwargs = data["parameters"]
    prompt_for_db_key = data["db_prompt"]

    if prompt_for_db_key not in list(prompt_for_db_dict_cache.keys()):
        prompt_for_db_dict_cache[prompt_for_db_key] = download_prompt_from_s3(prompt_for_db_key)
    else:
        print(f"{prompt_for_db_key} found in cache, {prompt_for_db_dict_cache.keys()}!")
    
    sample_prompt = copy.copy(prompt_for_db_dict_cache[prompt_for_db_key])
    sample_prompt = sample_prompt.format(question=text)
    
    outputs = predictor(sample_prompt, **generation_kwargs)
    result = outputs # [{"generated_text": outputs}]
    result = result[0]['generated_text'].strip().replace(';', '')
    result = sqlparse.format(result, reindent=True, keyword_case='upper')
    result = f"""%%sm_sql --metastore-id {prompt_for_db_key.split('.')[0]} --metastore-type GLUE_CONNECTION\n\n{result}\n"""
    result = [{'generated_text': result}]
    
    return Output().add(result)
