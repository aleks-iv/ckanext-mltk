import json
import logging

import openai

import ckanext.mltk.config as mltk_config

log = logging.getLogger(__name__)


def generate_data_with_chatgpt(text: str) -> dict:
    """Generates keywords, title, and description using a single ChatGPT request."""
    prompt = (
        "From the following text, perform the following tasks:\n"
        "1. Extract the top 10 keywords and present them as a list.\n"
        "2. Generate a concise and informative title.\n"
        "3. Provide a brief description.\n\n"
        "Provide ONLY the results in JSON format without any additional text or "
        "characters. The JSON should follow this structure:\n"
        "{\n"
        '  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", '
        '"keyword6", "keyword7", "keyword8", "keyword9", "keyword10"],\n'
        '  "title": "Title",\n'
        '  "description": "Description"\n'
        "}\n\n"
        f"Text:\n{text}"
    )

    response = _openai_chat_completion(
        prompt=prompt,
        system_message="You are an assistant that performs text analysis.",
        max_tokens=mltk_config.openai_max_tokens()["combined"],
        temperature=mltk_config.openai_temperature(),
    )
    return _parse_chatgpt_combined_response(response)


def _parse_chatgpt_combined_response(response: str) -> dict:
    """Parses the response from ChatGPT.

    The response should be in the following format:
        {
        "keywords": ["keyword1", "keyword2", ..., "keyword10"],
        "title": "Title",
        "description": "Description"
    }
    """
    data = {
        "keywords": [],
        "title": "Untitled",
        "description": "No description available.",
    }

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        log.exception("Failed to parse ChatGPT response as JSON.")

    return data


def _openai_chat_completion(
    prompt: str, system_message: str, max_tokens: int, temperature: float
) -> str:
    """Helper function to interact with OpenAI's ChatCompletion API."""
    client = openai.OpenAI(api_key=mltk_config.openai_api_key())

    completion = client.chat.completions.create(
        model=mltk_config.openai_model(),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temperature,
    )
    return completion.choices[0].message.content
