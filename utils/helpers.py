from datetime import datetime
from typing import List, Dict
from IPython.display import display, Markdown, HTML


def pretty_print_html(text):
    # Replace newline characters with <br> tags
    html_text = text.replace('\n', '<br>')
    # Apply HTML formatting
    html_formatted = f'<pre style="font-family: monospace; background-color: #f8f8f8; padding: 10px; border-radius: 5px; border: 1px solid #0077b6;">{html_text}</pre>'
    # Display the formatted HTML
    return HTML(html_formatted)


def set_meta_llama_params(
    max_new_tokens=512,
    top_p=0.9,
    temperature=0.6,
):
    """ set Llama parameters """
    llama_params = {}
    llama_params['max_new_tokens'] = max_new_tokens
    llama_params['top_p'] = top_p
    llama_params['temperature'] = temperature
    return llama_params


def print_dialog(inputs, payload, response):
    dialog_output = []
    for msg in inputs:
        dialog_output.append(f"**{msg['role'].upper()}**: {msg['content']}\n")
    dialog_output.append(f"**ASSISTANT**: {response['generated_text']}")
    dialog_output.append("\n---\n")
    
    display(Markdown('\n'.join(dialog_output)))

def format_messages(messages: List[Dict[str, str]]) -> List[str]:
    """
    Format messages for Llama 3+ chat models.
    
    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and 
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    # auto assistant suffix
    # messages.append({"role": "assistant"})
    
    output = "<|begin_of_text|>"
    # Adding the inferred prefix
    _system_prefix = f"\n\nCutting Knowledge Date: December 2023\nToday Date: {datetime.now().strftime('%d %b %Y')}\n\n"
    for i, entry in enumerate(messages):
        output += f"<|start_header_id|>{entry['role']}<|end_header_id|>"
        if i == 0:
            output += f"{_system_prefix}{entry['content']}<|eot_id|>"
        elif i >= 1 and 'content' in entry:
            output += f"\n\n{entry['content']}<|eot_id|>"
    output += "<|start_header_id|>assistant<|end_header_id|>\n"
    return output
