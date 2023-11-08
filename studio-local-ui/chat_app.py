from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools, AgentOutputParser, initialize_agent, ConversationalChatAgent, AgentExecutor
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
import streamlit as st
from sagemaker import session
import json
import re

f = open("endpoint_name.txt", "r")
endpoint_name = f.read()
f.close()

f = open("custom_attribute.txt", "r")
custom_attributes = f.read()
f.close()

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs):
        input_str = json.dumps({"inputs" : [[
        {"role" : "user", "content" : prompt}]],
        "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')
    
    def transform_output(self, output):
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

class OutputParser(AgentOutputParser):

    def parse(self, response):
        try:
            parsed_response=parse_json_markdown(response)
            step, step_input = parsed_response["step"], parsed_response["step_input"]
            if step == "Final Answer":
                return AgentFinish({"output": step_input}, response)
            else:
                return AgentAction(step, step_input, response)
        except:
            return AgentFinish({"output": response}, response)
        
    def get_format_instructions(self):
        return FORMAT_INSTRUCTIONS

parser = OutputParser()

st.set_page_config(page_title="LangChain: Chat with search and math", page_icon="🦜")
st.title("🦜 LangChain: Chat with search and math")

content_handler = ContentHandler()
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
parser = OutputParser()

llm=SagemakerEndpoint(
             endpoint_name=endpoint_name, 
             region_name=session.Session().boto_region_name, 
             model_kwargs={"max_new_tokens": 700, "top_p": 0.9, "temperature": 0.6},
             endpoint_kwargs={"CustomAttributes": custom_attributes},
             content_handler=content_handler
         )

tools = load_tools(["llm-math", "wikipedia"], llm=llm)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    memory=memory,
    llm=llm,
    tools=tools,
    verbose=True,
    agent_kwargs={
        "output_parser": parser
    }
)

system_message = system_message = """

<>\n Assistant is designed to build JSON and answer a wide variety of User questions.

Assistant must use JSON strings that contain "step" and "step_input" parameters. All of Assistant's communication is performed using this JSON format.

Tools available to Assistant are:

- "Wikipedia": Useful when you need a summary of a person, place, historical event, or other subject. Input is typically a noun, like a person, place, historical event, or another subject.
  - To use the wikipedia tool, Assistant should format the JSON like the following before getting the response and returning to the user:
    ```json
    {{"step": "Wikipedia",
      "step_input": "Statue of Liberty"}}
    ```
- "Calculator": Useful for when you need to answer questions about math. Input is one or more number combined with one or more math operations (addition, subtraction, multiplation, division, square root, exponetnial, and more).
  - To use the calculator tool, Assistant should format the JSON like the following so before getting the response and returning to the user:
    ```json
    {{"step": "Calculator",
      "step_input": "24*189"}}
    ```

Here is the set of previous interactions between the User and Assistant:

User: Hi!
Assistant: ```
{{"step": "Final Answer",
 "step_input": "Hello! How can I assist today?"}}
```
User: What is 9 cubed?
Assistant: ```
{{"step": "Calculator",
 "step_input": "9**3"}}
```
User: 729
Assistant: ```
{{"step": "Final Answer",
 "step_input": "The answer to your question is 729."}}
```
User: Can you tell me about the Statue of Liberty?
Assistant: ```
{{"step": "Wikipedia",
 "step_input": "Statue of Liberty"}}
```
User: The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City, in the United States. The copper statue, a gift from the people of France, was designed by French sculptor Frédéric Auguste Bartholdi and its metal framework was built by Gustave Eiffel.
Assistant: ```
{{"step": "Final Answer",
 "step_input": "Sure! The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City, in the United States. The copper statue, a gift from the people of France, was designed by French sculptor Frédéric Auguste Bartholdi and its metal framework was built by Gustave Eiffel."}}
```
User: What is the square root of 81?
Assistant: ```
{{"step": "Calculator",
 "step_input": "sqrt(81)"}}
```
User: 9
Assistant: ```
{{"step": "Final Answer",
 "step_input": "The answer to your question is 9."}}
```

Assistant should use a tool only if needed, but if the assistant does use a tool, the result of the tool must always be returned back to the user with a "Final Answer" step. Only use the calculator if the 'step_input' includes numbers. \n<>\n\n
"""

few_shot = agent.agent.create_prompt(
    system_message=system_message,
    tools=tools
)
agent.agent.llm_chain.prompt = few_shot

agent.agent.llm_chain.prompt.messages[2].prompt.template = "[INST] Respond in JSON with 'step' and 'step_input' values until you return an 'step': 'Final Answer', along with the 'step_input'. [/INST] \nUser: {input}"


if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    memory.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Please ask me a question!"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent(prompt, callbacks=[st_cb])["output"]
        response = re.sub("\{.*?\}","",response)
        st.write(response)