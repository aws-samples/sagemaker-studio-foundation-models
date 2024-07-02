from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
import boto3,json


def rank(input_path, output_path):
    bedrock_client = boto3.client('bedrock-runtime')
    model_kwargs = {
        "max_tokens": 120,
        "temperature": 0,
        "top_k": 250,
        "top_p": 0.999,
    }
    bedrock_model = BedrockChat(
        client=bedrock_client,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs=model_kwargs,
    )
    template = '''
    I'd like you to rank the responses to the question below based on how much they align with organisation's brand and values.
    Here is some information about the organisation's brand and values in <about></about> tags.
    <about>
    {context}
    </about>
    Here are some instructions you must follow:
    1. Rank each answer with an number between and 1 and 4. 1 is the most aligned and 4 the least aligned.
    2. Never assign the same number to more than one answers.
    3. Add your ranking to an array.
    4. Check that all the numbers in the array are unique and repeat the process if not
    Here is the question: <question>{question}</question>
    Here are the answers in <answers></answers> tags.
    <answers>
    {responses}
    </answers>
    Return the JSON in the response in line and nothing else . It should contain 
    an attribute called responseRankings with value the array from the step 3 above. 
    Think about your answer first before you respond.
    '''
    messages = [
        ("system", "You are a human evaluator. Your role is to rank responses to specific questions."),
        ("human", template),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | bedrock_model | StrOutputParser()
    with open(input_path, 'r') as file:
        json_string = file.read()
    data = json.loads(json_string)
    response = chain.batch(data, config={"max_concurrency": 5})
    json_list = [json.loads(item) for item in response]
    with open(output_path, 'w') as file:
        json.dump(json_list, file, indent=2)