from djl_python import Input, Output
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Any, Dict, Tuple
# import deepspeed
import warnings
import tarfile

predictor = None

def get_model(properties):
    print("CUDA devices", torch.cuda.device_count())
    print("Model Load Properties ===>", properties)
    cwd = properties["model_id"]
    print("Model Load Path ===>", cwd)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    print(f"Loading model from {cwd}")
    model = AutoModelForCausalLM.from_pretrained(
        cwd,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto"
    )
    print(f"Loading tokenizer from {cwd}")
    tokenizer = AutoTokenizer.from_pretrained(
        cwd, 
        padding_side="left",
        add_eos_token=True
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"Model loaded!!")

    return model, tokenizer


def handle(inputs: Input) -> None:
    global predictor, tokenizer

    instruction = "Pretend you are an expert financial analyst. You are provided with company financial document excerpt, you need read through the document and provide a short financial summary from the excerpt."
    
    if not predictor:
        predictor, tokenizer = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
        
    data = inputs.get_as_json()

    print(f"====================> Recevied request", data)
    
    text = data["input"]
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text}
    ]

    input_inference_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(predictor.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    generation_kwargs = data["properties"]
    generation_kwargs["eos_token_id"] = terminators

    print(f"generator kwargs===================>", generation_kwargs)
    
    outputs = predictor.generate(input_inference_ids, **generation_kwargs)
    response = outputs[0][input_inference_ids.shape[-1]:]
    llm_response = tokenizer.decode(response, skip_special_tokens=True)
    print("\n LLM Response ===============> ", llm_response)
    
    return Output().add({"generated_text": llm_response})