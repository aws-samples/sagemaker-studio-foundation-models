import argparse
import pyfiglet
import pprint
import sagemaker
from sagemaker.estimator import Estimator


pp = pprint.PrettyPrinter(indent=4)


ascii_banner = pyfiglet.figlet_format("Welcome to SageMaker Studio GenAI Workshop!!")
print(ascii_banner)


# Create a SageMaker session
sess = sagemaker.Session()


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Example script to demonstrate argparse usage.")

    parser.add_argument(
        "--role", 
        type=str, 
        help="SageMaker Execution Role", 
    )

    parser.add_argument(
        "--endpoint_name", 
        type=str, 
        default="ft-meta-llama2-13b-neuron-chat-tg-ep",
        help="Fine-Tuned Model deploy endpoint name", 
    )

    parser.add_argument(
        "--instance_type", 
        default="ml.inf2.24xlarge",
        type=str, 
        help="Endpoint Instance Type", 
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


def deploy(user_args):

    # Get the SageMaker client from the current session
    sagemaker_client = sess.sagemaker_client

    # List training jobs
    response = sagemaker_client.list_training_jobs()

    # Iterating through the training jobs and printing their names
    for job in response['TrainingJobSummaries']:
        if job['TrainingJobStatus'] == 'Completed':
            training_job_name = job['TrainingJobName']
            break
    
    print(f"Found Training Job  ---> {training_job_name} \n")

    estimator = Estimator.attach(
        training_job_name=training_job_name, 
        sagemaker_session=sess
    )

    # deploy model
    finetuned_predictor = estimator.deploy(
        endpoint_name=user_args.endpoint_name,
        instance_type=user_args.instance_type,
        initial_instance_count=1,
        wait=True
    )


if __name__ == "__main__":

    args = parse_args()

    print("\n-------------- Runtime Arguments------------")
    pp.pprint(vars(args))
    print("--------------------------------------------\n")

    deploy(user_args=args)

