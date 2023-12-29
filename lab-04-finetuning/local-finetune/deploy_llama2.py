"""
python3 deploy_llama2.py \
    --sm_endpoint_name=ft-meta-llama2-7b-chat-tg-ep \
    --sm_model_name=ft-meta-llama2-7b-chat-tg-model \
    --instance_type=ml.g5.2xlarge \
    --hf_model_id=Mikael110/llama-2-7b-guanaco-fp16
"""

import os
import boto3
import argparse
import pyfiglet
import pprint
from datetime import datetime
import botocore
import sagemaker
import torch
from sagemaker import image_uris
from sagemaker.model import Model
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


pp = pprint.PrettyPrinter(indent=4)


ascii_banner = pyfiglet.figlet_format("Welcome to SageMaker Studio GenAI Workshop!!")
print(ascii_banner)


sts_client = botocore.session.Session().create_client("sts")
role_arn = sts_client.get_caller_identity().get("Arn")
# conver assumed role to just role arn
role = role_arn.replace('assumed-role', 'role').replace('/SageMaker', '').replace('sts', 'iam')
print(f"Using role ---> ", role)


# Create a SageMaker session
sess = sagemaker.Session()
region = sagemaker.Session().boto_region_name
bucket = sess.default_bucket() 
sm_client = boto3.client('sagemaker', region_name=region)
smr_client = boto3.client("sagemaker-runtime")


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Deploy our fine-tuned model to a sagemaker endpoint")

    parser.add_argument(
        "--sm_endpoint_name", 
        type=str, 
        default="ft-meta-llama2-13b-neuron-chat-tg-ep",
        help="Fine-Tuned Model deploy endpoint name", 
    )

    parser.add_argument(
        "--sm_model_name", 
        type=str, 
        default="ft-meta-llama2-7b-chat-tg-model",
        help="Fine-Tuned Model deploy model name", 
    )

    parser.add_argument(
        "--instance_type", 
        default="ml.inf2.24xlarge",
        type=str, 
        help="Endpoint Instance Type", 
    )

    parser.add_argument(
        "--hf_model_id", 
        type=str, 
        help="HuggingFace model id", 
        default="Mikael110/llama-2-7b-guanaco-fp16"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def merge_base_with_lora(model_temp_local_dir, model_dest_suffix):

    tokenizer = AutoTokenizer.from_pretrained(model_temp_local_dir)

    # load PEFT model in fp16
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_temp_local_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )  

    print("\nMerging Base and LoRA!")
    # Merge LoRA and base model and save
    model = model.merge_and_unload()

    merged_model_local_dir = os.path.join(model_temp_local_dir, "merged-weights")

    model.save_pretrained(
        merged_model_local_dir, 
        safe_serialization=True, 
        max_shard_size="2GB"
    )

    tokenizer.save_pretrained(
        merged_model_local_dir, 
        from_pt=True
    )

    pretrained_s3_model_uri = f"s3://{bucket}/models/{model_dest_suffix}"

    print(f"\nUploading model to {pretrained_s3_model_uri}")
    sagemaker.s3.S3Uploader.upload(
        merged_model_local_dir, 
        pretrained_s3_model_uri
    )

    return pretrained_s3_model_uri

def deploy(user_args):

    pretrained_s3_model_uri = merge_base_with_lora(
        os.path.join(user_args.hf_model_id.replace('/', '-'), 'lora-finetune-dir'), 
        user_args.hf_model_id
    )

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    # create and write serving properties file
    serving_properties = f"""
    engine = MPI
    option.tensor_parallel_degree = 1
    option.rolling_batch = auto
    option.max_rolling_batch_size = 64
    option.model_loading_timeout = 3600
    option.paged_attention = true
    option.trust_remote_code = true
    option.dtype = fp16
    option.rolling_batch=lmi-dist
    option.max_rolling_batch_prefill_tokens=1560
    option.s3url = {pretrained_s3_model_uri}
    """.rstrip()

    with open(os.path.join(f"./{model_dir}", "serving.properties"), "w") as prop_file:
        prop_file.write(serving_properties)
    
    os.system(f"tar czvf {model_dir}.tar.gz ./{model_dir}/")

    inference_image_uri = image_uris.retrieve(
        framework="djl-deepspeed", 
        region=region, 
        version="0.23.0"
    )
    print(f"Image going to be used is ---- > {inference_image_uri}")

    s3_code_prefix = "model/artifacts"

    code_artifact = sess.upload_data(f"{model_dir}.tar.gz", bucket, s3_code_prefix)
    print(f"S3 Code or Model tar ball uploaded to --- > {code_artifact}")

    ft_model = Model(
        sagemaker_session=sess,
        image_uri=inference_image_uri,
        model_data=code_artifact,
        role=role,
        name=user_args.sm_model_name,
    )

    print(f"Deploying model with endpoint name {user_args.sm_endpoint_name}")
    ft_model.deploy(
        initial_instance_count=1,
        instance_type=user_args.instance_type,
        endpoint_name=user_args.sm_endpoint_name,
        container_startup_health_check_timeout=900,
        wait=True, # <-- Set to True, if you would prefer to wait for the endpoint to spin up
    )

    return 0


if __name__ == "__main__":

    args = parse_args()

    print("\n-------------- Runtime Arguments------------")
    pp.pprint(vars(args))
    print("--------------------------------------------\n")

    deploy(user_args=args)

