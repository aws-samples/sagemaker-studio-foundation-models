"""
python3 fine_tune_llama2_trn1.py \
    --dataset_name=databricks/databricks-dolly-15k \
    --js_hf_model_id=meta-textgenerationneuron-llama-2-13b \
    --js_hf_model_version=1.* \
    --max_steps=2 \
    --lr=0.0001 \
    --batch_size=1000 \
    --instance_type=ml.trn1.32xlarge
"""

import os
import uuid
import argparse
import pyfiglet
import pprint
import botocore
import sagemaker
import boto3
import json
from sagemaker.s3 import S3Uploader
from datasets import load_dataset
from sagemaker import hyperparameters
from sagemaker.jumpstart.estimator import JumpStartEstimator


pp = pprint.PrettyPrinter(indent=4)


llama2_eula_map = {'y': (True, "true"), 'n': (False, 'false')}


PROMPT = (
    """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}### Response:\n{response}\n\n<s>"""
)


# temporary workaround until reinvent
os.environ.update({
    "AWS_JUMPSTART_CONTENT_BUCKET_OVERRIDE": "jumpstart-cache-alpha-us-west-2",
    "AWS_JUMPSTART_GATED_CONTENT_BUCKET_OVERRIDE": "jumpstart-private-cache-prod-us-west-2",
    "AWS_DATA_PATH": "/home/sagemaker-user/models"
})

ascii_banner = pyfiglet.figlet_format("Welcome to SageMaker Studio GenAI Workshop!!")
print(ascii_banner)


session_uuid = uuid.uuid4()

sts_client = botocore.session.Session().create_client("sts")
role_arn = sts_client.get_caller_identity().get("Arn")
# conver assumed role to just role arn
role = role_arn.replace('assumed-role', 'role').replace('/SageMaker', '')

print(f"Using role ---> ", role)


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Example script to demonstrate argparse usage.")

    parser.add_argument(
        "--dataset_s3_dest", 
        type=str, 
        default=None,
        help="Dataset S3 destination, ex: s3://my-example-bucket/my-prefix/, set default to use sagemaker bucket",
    )

    parser.add_argument(
        "--dataset_name", 
        type=str, 
        help="HuggingFace dataset name, ex: databricks/databricks-dolly-15k", 
        default="databricks/databricks-dolly-15k"
    )
    
    parser.add_argument(
        "--js_hf_model_id", 
        type=str, 
        help="JumpStart HuggingFace model id", 
        default="meta-textgenerationneuron-llama-2-13b"
    )

    parser.add_argument(
        "--js_hf_model_version", 
        type=str, 
        help="JumpStart HuggingFace model version", 
        default="1.*"
    )

    parser.add_argument(
        "--max_steps", 
        type=str, 
        help="Max training steps", 
        default="25"
    )

    parser.add_argument(
        "--lr", 
        type=str, 
        help="Fine-tuning model learning rate", 
        default="0.0001"
    )

    parser.add_argument(
        "--batch_size", 
        type=str, 
        help="Global model batch size, this is per accelerator batch size * # of accelerators", 
        default="1000"
    )

    parser.add_argument(
        "--instance_type", 
        type=str, 
        help="Training Instance Type", 
        default="ml.trn1.32xlarge"
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


def apply_prompt_template(sample):
    return {
        "text": PROMPT.format(
            instruction=sample["instruction"], 
            context=sample["context"], 
            response=sample["response"]
        )
    }


def upload_to_s3_uri(destination_s3_uri, task):

    local_data_file = f"dolly/processed-train-{task}.jsonl"

    if destination_s3_uri is None:
        output_bucket = sagemaker.Session().default_bucket()
        destination_s3_uri = f"s3://{output_bucket}/fine-tuning/{session_uuid}/dolly_dataset"

    S3Uploader.upload(local_data_file, destination_s3_uri)

    print(f"Training data  ---> : {destination_s3_uri}")

    return destination_s3_uri


def prepare_dataset(dataset_name, destination_s3_uri):

    dolly_dataset = load_dataset(
        dataset_name, 
        split="train[:10%]"
    )

    task = "information_extraction"

    # To train for summarization/closed question and answering, you can replace the assertion in next line to example["category"] == "sumarization"/"closed_qa".
    summarization_dataset = dolly_dataset.filter(
        lambda example: example["category"] == task
    )
    summarization_dataset = summarization_dataset.remove_columns("category")

    # We split the dataset into two where test data is used to evaluate at the end.
    train_and_test_dataset = summarization_dataset.train_test_split(test_size=0.1)

    # Dumping the training data to a local file to be used for training.
    train_and_test_dataset["train"].to_json("train.jsonl")

    print("Dataset Sample:\n")
    pp.pprint(train_and_test_dataset["train"][-1])

    dataset_processed = train_and_test_dataset.map(
        apply_prompt_template, 
        remove_columns=list(train_and_test_dataset["train"].features)
    )

    dataset_processed["train"].to_json(f"dolly/processed-train-{task}.jsonl")
    dataset_processed["test"].to_json(f"dolly/processed-test-{task}.jsonl")

    print("uploading dataset to s3")
    dataset_s3_upload_path = upload_to_s3_uri(
        destination_s3_uri=destination_s3_uri, 
        task=task
    )

    return dataset_s3_upload_path
    


def fine_tune(user_args, llama_eula):

    destination_s3_uri = prepare_dataset(
        dataset_name=user_args.dataset_name, 
        destination_s3_uri=user_args.dataset_s3_dest
    )

    sess_hyperparams = hyperparameters.retrieve_default(
        model_id=user_args.js_hf_model_id, 
        model_version=user_args.js_hf_model_version
    )

    hyperparameters.validate(
        model_id=user_args.js_hf_model_id, 
        model_version=user_args.js_hf_model_version, 
        hyperparameters=sess_hyperparams
    )
    sess_hyperparams["max_steps"] = user_args.max_steps
    sess_hyperparams["learning_rate"] = user_args.lr
    sess_hyperparams["global_train_batch_size"] = user_args.batch_size

    print("\n-------------- Fine-Tuning Hyperparameters------------")
    pp.pprint(sess_hyperparams)
    print("\n")


    estimator = JumpStartEstimator(
        model_id=user_args.js_hf_model_id,
        model_version=user_args.js_hf_model_version,
        hyperparameters=sess_hyperparams,
        environment={"accept_eula": llama_eula}, 
        role=role
    )

    estimator.fit(
        {
            "train": destination_s3_uri
        },
        wait=True
    )

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
