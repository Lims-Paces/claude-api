from dotenv import loaddotenv
loaddotenv()

import os
from anthropic import Anthropic
from anthropic.types import MessageParam


client = Anthropic(apikey=os.environ.get('ANTHROPICAPI_KEY'))

def estimate_token(prompt):
    count = client.count_tokens(prompt)
    return count

def submit_prompt(prompt, system_prompt):
    with client.messages.stream(
        model='claude-3-5-sonnet-20240620',
        system=system_prompt,
        max_tokens=1024,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    ) as stream:
        try:
            for text in stream.text_stream:
                print(text, end='', flush=True)
        except Exception as e:
            print(e)
            raise e
    print('j')

if __name__ == "__main":
    prompt = input('Enter your prompt: ')
    system_prompt = """
    You are a customer service agent tasked with classifying emails by type. Please output your answer and then justify your classification.

    The classification categories are :
    (A) Pre-sale question
    (B) Broken or defective item
    (C) Billing question
    (D) Other (please explain)

    How would you classify this email
"""
    print(f'Tokens for this request: {estimate_token(prompt)}')
    submit_prompt(prompt, system_prompt=system_prompt)