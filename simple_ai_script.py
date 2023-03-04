import json
import openai
from typing import Iterator, List, Dict
from csv import DictReader, DictWriter

API_KEY = "TOKEN_HERE"

MODEL_NAME = "text-davinci-003"
MODEL_BASE_TEMPERATURE = 0.7
MODEL_BASE_MAX_TOKENS = 512
MODEL_BASE_TOP_P = 1
MODEL_BASE_FREQUENCY_PENALTY = 0
MODEL_BASE_PRESENCE_PENALTY = 0


def read_messages_from_file(file_name: str) -> Iterator[Dict]:
    with open(file_name) as in_file:
        yield from DictReader(in_file)


def process_message(openai_key: str, prompt: str) -> str:
    openai.api_key = openai_key
    response = openai.Completion.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=MODEL_BASE_TEMPERATURE,
        max_tokens=MODEL_BASE_MAX_TOKENS,
        top_p=MODEL_BASE_TOP_P,
        frequency_penalty=MODEL_BASE_FREQUENCY_PENALTY,
        presence_penalty=MODEL_BASE_PRESENCE_PENALTY
    )
    return response.choices[0].text


def generate_prompt(base_prompt: str, message: str) -> str:
    return f"{base_prompt}\n\"\"\"{message}\"\"\""


def analyze_messages_in_file(in_file_name: str,
                             text_column: str,
                             out_file_name: str,
                             out_file_columns: List[str],
                             base_prompt: str,
                             openai_key: str) -> None:
    with open(out_file_name, 'w') as out_file:
        writer = DictWriter(out_file, out_file_columns)
        for message in read_messages_from_file(in_file_name):
            prompt = generate_prompt(base_prompt, message.get(text_column))
            data = json.loads(process_message(openai_key, prompt))
            writer.writerow({**message, "open_ai_output": data})


if __name__ == '__main__':
    base_prompt = """
    You are an application for analyzing text. I need a response from you in the following format:
    {
    "tone" : tone,
    "grammatically_correct" : grammatically_correct,
    "message_clarity" : message_clarity
    "message_language" : "message_language"
    "to_respond" : to_respond
    }

    where the tone is a category response of one of the following ["very positive", "positive", "neutral", 
    "slightly negative", "negative", "very negative"]
    where grammatically_correct is a boolean response, true if the text is grammatically correct, false if it is not
    where message clarity is a category response of one of the following ["clear", "slightly clear", "not clear"]
    where message_language is the langauge the message was written in
    to_respond is a true or false value, it is true if the text contains some feedback, or is written in a constructive 
    way, it is also true if it is a compliment or feedback on a concrete thing besides some text or post,
    it is false if the message is a generic praise or is just a vague complaint without any question or action to 
    take. 

    The message to analyze is :

    """
    analyze_messages_in_file(in_file_name="messages.csv",
                             text_column="message",
                             out_file_name="output.csv",
                             out_file_columns=["id", "message", "open_ai_output"],
                             base_prompt=base_prompt,
                             openai_key=API_KEY)
