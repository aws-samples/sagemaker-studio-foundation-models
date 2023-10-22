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

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs" : [[
        {"role" : "user", "content" : prompt}]],
        "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

class OutputParser(AgentOutputParser):

    def parse(self, text: str):
        try:
            parsed=parse_json_markdown(text)
            action, action_input = parsed["action"], parsed["action_input"]
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)
        except:
            return AgentFinish({"output": text}, text)
        
    @property
    def _type(self) -> str:
        return "conversational_chat"
        
    def get_format_instructions(self):
        return FORMAT_INSTRUCTIONS
    

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
             endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
             content_handler=content_handler
         )

tools = load_tools(["llm-math", "wikipedia"], llm=llm)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    memory=memory,
    agent_kwargs={
        "output_parser": parser
    }
)

system_message = """

<>\n Assistant is a JSON builder designed to assist with a wide range of tasks.

Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

All of Assistant's communication is performed using this JSON format.

Tools available to Assistant are:

- "Wikipedia": Useful when you need a summary of a person, place, company, historical event, or other subject. Input is typically a noun, like a person, place, company, historical event, or other subject.
  - To use the wikipedia tool, Assistant should write like so before getting the response and returning to the user:
    ```json
    {{"action": "Wikipedia",
      "action_input": "Statue of Liberty"}}
    ```
- "Calculator": Useful for when you need to answer questions about math. Only use this if the input would contain numbers.
  - To use the calculator tool, Assistant should write like so before getting the response and returning to the user:
    ```json
    {{"action": "Calculator",
      "action_input": "sqrt(9)"}}
    ```

Here are some previous conversations between the Assistant and User:

User: Hey how are you doing?
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "I'm good thanks, how are you?"}}
```
User: What is the square root of 16?
Assistant: ```json
{{"action": "Calculator",
 "action_input": "sqrt(16)"}}
```
User: 2.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 2."}}
```
User: Can you tell me 4 to the power of 2?
Assistant: ```json
{{"action": "Calculator",
 "action_input": "4**2"}}
```
User: 16.0
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "It looks like the answer is 16."}}
```
User: Can you tell me about the Statue of Liberty?
Assistant: ```json
{{"action": "Wikipedia",
 "action_input": "Statue of Liberty"}}
```
User: The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City, in the United States. The copper statue, a gift from the people of France, was designed by French sculptor Frédéric Auguste Bartholdi and its metal framework was built by Gustave Eiffel.
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "Sure! The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City, in the United States. The copper statue, a gift from the people of France, was designed by French sculptor Frédéric Auguste Bartholdi and its metal framework was built by Gustave Eiffel."}}
```

Assistant should use a tool only if needed, but if the assistant does use a tool, the result of the tool must always be returned back to the user with a "Final Answer" format. Only use the calculator if the 'action_input' includes numbers. \n<>\n\n
"""

zero_shot = agent.agent.create_prompt(
    system_message=system_message,
    tools=tools
)
agent.agent.llm_chain.prompt = zero_shot

agent.agent.llm_chain.prompt.messages[2].prompt.template = "[INST] Respond in JSON with 'action' and 'action_input' values until you return an 'action': 'final answer', along with the 'action_input'. [/INST] \nUser: {input}"


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