"""
python3 local_fine_tune_llama2.py \
    --dataset_name=databricks/databricks-dolly-15k \
    --hf_model_id=Mikael110/llama-2-7b-guanaco-fp16 \
    --epochs=1 \
    --per_device_train_batch_size=1 \
    --per_device_valid_batch_size=1 \
    --logging_steps=1
"""

import os
import pprint
import argparse
import botocore
import boto3
import sagemaker
from datasets import load_dataset
from random import randrange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from datasets import load_from_disk
import torch
import bitsandbytes as bnb
import sagemaker
import shutil
import pyfiglet
from random import randint
from itertools import chain
from functools import partial
from transformers import AutoTokenizer


llama2_eula_map = {'y': (True, "true"), 'n': (False, 'false')}

pp = pprint.PrettyPrinter(indent=4)

ascii_banner = pyfiglet.figlet_format("Welcome to SageMaker Studio GenAI Workshop!!")
print(ascii_banner)

sts_client = botocore.session.Session().create_client("sts")
role_arn = sts_client.get_caller_identity().get("Arn")
role = role_arn.replace('assumed-role', 'role').replace('/SageMaker', '').replace('sts', 'iam')

print(f"Using role ---> ", role)

sess = sagemaker.Session()
region = sagemaker.Session().boto_region_name
bucket = sess.default_bucket() 
sm_client = boto3.client('sagemaker', region_name=region)
smr_client = boto3.client("sagemaker-runtime")


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Local fine-tune Llama2 from HuggingFace hub")

    parser.add_argument(
        "--dataset_name", 
        type=str, 
        help="HuggingFace dataset name, ex: databricks/databricks-dolly-15k", 
        default="databricks/databricks-dolly-15k"
    )
    
    parser.add_argument(
        "--hf_model_id", 
        type=str, 
        help="HuggingFace model id", 
        default="Mikael110/llama-2-7b-guanaco-fp16"
    )

    parser.add_argument("--epochs", type=int, default=1, help="No of epochs to train the model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size to use for training.")
    parser.add_argument("--per_device_valid_batch_size", type=int, default=2, help="Batch size to use for validation.")   
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Model learning rate")
    parser.add_argument("--seed", type=int, default=8, help="Seed to use for training")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Path to deepspeed config file")
    parser.add_argument("--lora_r", type=int, default=64, help="Lora attention dimension value")
    parser.add_argument("--lora_alpha", type=int, default=16, help="The alpha parameter for Lora scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="The dropout probability for Lora layers")
    parser.add_argument(
        "--task_type", 
        type=str, default="CAUSAL_LM", 
        help="Choose from: CAUSAL_LM, FEATURE_EXTRACTION, QUESTION_ANS, SEQ_2_SEQ_LM, SEQ_CLS, TOKEN_CLS"
    )
    parser.add_argument("--logging_steps", type=int, default=1, help="Step interval to start logging to console/sagemaker experiments")

    # Parse the arguments
    args = parser.parse_args()

    return args


def format_dolly(sample, incl_answer=True):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}" if incl_answer else None
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])

    if not incl_answer:
        return prompt, sample['response']
    else:
        return prompt


# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample


def prepare_dataset(dataset_name, tokenizer):

    # Load dataset from the hub
    train_dataset = load_dataset(dataset_name, split="train[:100]")
    validation_dataset = load_dataset(dataset_name, split="train[-25:]")

    print(f"Training size: {len(train_dataset)} | Validation size: {len(validation_dataset)}")
    print("\nTraining sample:\n")
    pp.pprint(train_dataset[randrange(len(train_dataset))])
    print("\nValidation sample:\n")
    pp.pprint(validation_dataset[randrange(len(validation_dataset))])

    # apply prompt template per sample
    # train
    train_dataset = train_dataset.map(template_dataset, remove_columns=list(train_dataset.features))
    # validation
    validation_dataset = validation_dataset.map(template_dataset, remove_columns=list(validation_dataset.features))
    # print random sample
    print(validation_dataset[randint(0, len(validation_dataset))]["text"])

    # empty list to save remainder from batches to use in next batch
    remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    def chunk(sample, remainder, chunk_length=2048):
        
        # Concatenate all texts and add remainder from previous batch
        concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
        concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
        # get total number of tokens for batch
        batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

        # get max number of chunks for batch
        if batch_total_length >= chunk_length:
            batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
            for k, t in concatenated_examples.items()
        }
        # add remainder to global variable for next batch
        remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
        # prepare labels
        result["labels"] = result["input_ids"].copy()
        return result
    
    # training
    lm_train_dataset = train_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(train_dataset.features)
    ).map(
        partial(chunk, remainder=remainder, chunk_length=1024),
        batched=True,
    )

    # validation
    lm_valid_dataset = validation_dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(validation_dataset.features)
    ).map(
        partial(chunk, remainder=remainder, chunk_length=1024),
        batched=True,
    )

    # Print total number of samples
    print(f"Total number of samples ---> Train: {len(lm_train_dataset)}, Validation: {len(lm_valid_dataset)}")

    return lm_train_dataset, lm_valid_dataset
    

def download_huggingface_model(model_id):
    """
    Downloads model/weights from HuggingFace to local instance 
    and pins the model to GPU(s).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    return model


# Reference: https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    """Find all tuneable names for the model"""
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def fine_tune(user_args, llama_eula):

    set_seed(args.seed)

    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(user_args.hf_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, valid_dataset = prepare_dataset(
        dataset_name=user_args.dataset_name, 
        tokenizer=tokenizer
    )

    model = download_huggingface_model(model_id=user_args.hf_model_id)

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.gradient_checkpointing_enable()

    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    # configure model for Peft based fine-tuning
    peft_config = LoraConfig(
        r=user_args.lora_r,
        lora_alpha=user_args.lora_alpha,
        target_modules=modules,
        lora_dropout=user_args.lora_dropout,
        bias="none",
        task_type=user_args.task_type
    )
    model = get_peft_model(model, peft_config)

    print(
        f"Beginning to train model for Epochs:{user_args.epochs}, Train/Val Batch sizes:{user_args.per_device_train_batch_size}/{user_args.per_device_valid_batch_size}, Learning Rate:{user_args.learning_rate}, Logging Step: {user_args.logging_steps}"
    )

    # instantiate training args
    training_args = TrainingArguments(
        output_dir=f"{user_args.hf_model_id.replace('/', '-')}/outputs",
        per_device_train_batch_size=user_args.per_device_train_batch_size,
        per_device_eval_batch_size=user_args.per_device_valid_batch_size,
        bf16=False,  
        learning_rate=user_args.learning_rate,
        num_train_epochs=user_args.epochs,
        gradient_checkpointing=True,
        logging_dir=f"{user_args.hf_model_id.replace('/', '-')}/logs",
        logging_strategy="steps",
        logging_steps=user_args.logging_steps,
        save_strategy="no",
        report_to="tensorboard"
    )
    # instantiate a new trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator
    )
    model.config.use_cache = False 

    # train model
    trainer.train()

    # evaluate and return the metrics
    metrics = trainer.evaluate()

    print("\nEvaluation metrics ===>\n")
    pp.pprint(metrics)

    model_temp_local_dir = f"{user_args.hf_model_id.replace('/', '-')}/lora-finetune-dir/"

    trainer.model.save_pretrained(
        model_temp_local_dir, 
        safe_serialization=False
    )

    del model
    del trainer
    torch.cuda.empty_cache()

    tokenizer.save_pretrained(
        model_temp_local_dir, 
        from_pt=True
    )
    
    print(f"Model: {user_args.hf_model_id} saved to {model_temp_local_dir}")
    
    return 0


if __name__ == "__main__":
    args = parse_args()

    print("\n-------------- Runtime Arguments------------")
    pp.pprint(vars(args))
    print("--------------------------------------------\n")

    user_eula = input("Accept Llama 2 EULA? [y/n]: ")
    user_eula = user_eula.lower()

    assert user_eula in ['y', 'n'], "Invalid EULA response, please answer y or n"

    if llama2_eula_map[user_eula][0]:
        fine_tune(
            user_args=args, 
            llama_eula=llama2_eula_map[user_eula][-1]
        )
    else:
        print(f"Llama 2 EULA set to {user_eula}/{llama2_eula_map[user_eula][-1]}, aborting!")
