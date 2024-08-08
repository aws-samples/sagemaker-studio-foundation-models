from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain.agents import load_tools, AgentOutputParser, initialize_agent, ConversationalChatAgent, AgentExecutor
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
import streamlit as st
from sagemaker import session
import json
import re
import subprocess
from typing import Dict, List
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from transformers import AutoTokenizer
from datasets import load_dataset


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", 
    padding_side="left",
    add_eos_token=True
)
tokenizer.pad_token = tokenizer.eos_token


dataset_name = "mrSoul7766/ECTSum"
inference_dataset = load_dataset(
    dataset_name, 
    split="test"
).rename_column("text", "context")
inference_dataset = list(inference_dataset)


def llama3_prompt(text_blob, test=False):

    instruction = "Pretend you are an expert financial analyst. You are provided with company financial document excerpt, you need read through the document and provide a short financial summary from the excerpt."

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text_blob}
    ]

    input_chat_prompt = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=False
    )

    return input_chat_prompt

endpoint_name_preft = "pre-finetuned-llama3-8b-instruct"
endpoint_name_post = "post-finetuned-llama3-8b-instruct-develop-v1"

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs):
        optz_input = llama3_prompt(prompt)
        print("=================>", st.session_state.endpoint)
        print("=================>", st.session_state.model_kwargs)
        if 'pre' in st.session_state.endpoint:
            payload = {
                "inputs" : optz_input, 
                "parameters" : {**model_kwargs}
            }
        elif 'post' in st.session_state.endpoint:
            payload = {
                "input" : prompt, 
                "properties" : {**model_kwargs}
            }
        input_str = json.dumps(payload)
        return input_str.encode('utf-8')
    
    def transform_output(self, output):
        response_json = json.loads(output.read().decode("utf-8"))
        print("response_json", response_json)
        return response_json["generated_text"].replace('$', '\$')

st.set_page_config(page_title="LangChain: Fin-Summary with Llama 3 Model", page_icon="🦜")
st.title("🦜 LangChain: Chat with Finance Llama 3 Model")

content_handler = ContentHandler()

# Initialize Streamlit session state if not already initialized
if "endpoint" not in st.session_state:
    st.session_state.endpoint = endpoint_name_preft

# Add dropdown for endpoint selection
endpoint_option = st.sidebar.selectbox(
    "Select Model Endpoint",
    [f"Pre-Finetuned ({endpoint_name_preft})", f"Post-Finetuned ({endpoint_name_post})"],
    index=0 if st.session_state.endpoint == endpoint_name_preft else 1
)

# Update session state based on selection
if endpoint_option == f"Pre-Finetuned ({endpoint_name_preft})":
    st.session_state.endpoint = endpoint_name_preft
    st.session_state.model_kwargs = {
        "max_new_tokens": 128, 
        "top_p": 0.7, 
        "temperature": 0.6, 
        "stop": '<|eot_id|>',
        "do_sample": False
    }
else:
    st.session_state.endpoint = endpoint_name_post
    st.session_state.model_kwargs = {
        "max_new_tokens": 128, 
        "top_p": 0.7, 
        "temperature": 0.6,
        "do_sample": False
    }

# Add sliders for Temperature, Top p, and Max New Tokens
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.6, 0.1)
top_p = st.sidebar.slider("Top p", 0.0, 0.99, 0.7, 0.1)
max_new_tokens = st.sidebar.slider("Max New Tokens", 64, 512, 128, 16)

# Function to reinitialize the conversation agent with new parameters
def reinitialize_agent():
    st.session_state.model_kwargs.update({"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens})
    st.experimental_rerun()  # Rerun the script to reset the UI

# Submit button
if st.sidebar.button("Submit"):
    reinitialize_agent()

# Reinitialize the LLM with the selected endpoint
llm = SagemakerEndpoint(
    endpoint_name=st.session_state.endpoint, 
    region_name=session.Session().boto_region_name, 
    model_kwargs=st.session_state.model_kwargs,
    content_handler=content_handler
)

template = "{input}"

PROMPT = PromptTemplate(input_variables=["input"], template=template)

agent = LLMChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
)

if prompt := st.chat_input(placeholder="Please ask me a question!"):
    st.chat_message("user").write(prompt.replace('$', '\$'))
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(prompt, callbacks=[st_cb])
        response = re.sub("\{.*?\}","",response)
        st.write(response)
        # Adding a collapsible section with custom text
        with st.expander("Ground Truth (experimental)"):
            gt_ = None
            for i in inference_dataset:
                if i['context'] == prompt:
                    gt_ = i['summary']
                    break
            if gt_ is None:
                gt_ = "No GT found!"
            gt_ = gt_.replace('$', '\$')
            st.write(f"{gt_}")
