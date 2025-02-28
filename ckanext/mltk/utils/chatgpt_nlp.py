from __future__ import annotations

import json
import logging

import openai
from openai.types.chat import ChatCompletionMessage

import ckanext.mltk.config as mltk_config

log = logging.getLogger(__name__)


def generate_data_with_chatgpt(
    text: str, entity: str = "dataset"
) -> dict[str, str | list[str]]:
    """Generates keywords, title, and description using ChatGPT.

    Args:
        text: Input text to analyze.
        entity: Type of entity ('dataset' or 'resource').

    Returns:
        Dict with 'keywords' (list), 'title' (str), 'description' (str).

    Raises:
        openai.OpenAIError: If API call fails.
    """
    system_message = (
        f"You are an assistant for a CKAN open data portal. For the given text, "
        f"generate: 1) Top 10 keywords (list), 2) A concise title for the {entity}, "
        f"3) A description for the {entity}. Return results in JSON format."
    )
    prompt = f"Text: ```{text[:4096]}```"  # Truncate to avoid token limits

    functions = [
        {
            "name": "analyze_text",
            "description": f"Extract keywords, title, and description for a {entity}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["keywords", "title", "description"],
            },
        }
    ]

    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        response = _get_response(messages, functions)
        return _parse_chatgpt_response(response)
    except openai.OpenAIError:
        log.exception("ChatGPT API error")
        return {"keywords": [], "title": "", "description": ""}


def check_if_dataset_description_is_relevant(
    dataset_description: str, resources_text: list[str]
) -> bool:
    """Checks if the dataset description is relevant to the resources content.

    Args:
        dataset_description: Existing dataset description.
        resources_text: List of resource texts.

    Returns:
        True if relevant, False otherwise.
    """
    system_message = (
        "You are an assistant for a CKAN open data portal. Determine if the dataset "
        "description is relevant to the resources content."
    )
    resources_str = " ".join(resources_text)[:4096]  # Truncate for safety
    prompt = (
        f"Description: ```{dataset_description}```\nResources: ```{resources_str}```"
    )

    functions = [
        {
            "name": "analyze_text",
            "description": "Check description relevance.",
            "parameters": {
                "type": "object",
                "properties": {"is_relevant": {"type": "boolean"}},
                "required": ["is_relevant"],
            },
        }
    ]

    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        response = _get_response(messages, functions)
        return _parse_chatgpt_response(response).get("is_relevant", False)
    except openai.OpenAIError:
        log.exception("ChatGPT relevance check error")
        return False


def _get_response(
    messages: list[dict[str, str]], functions: list[dict[str, str]] | None = None
) -> ChatCompletionMessage:
    """Gets the response from OpenAI's ChatCompletion API.

    Args:
        messages: List of message dictionaries.
        functions: Optional list of function definitions for structured output.

    Returns:
        ChatCompletionMessage object.

    Raises:
        openai.OpenAIError: If API call fails.
    """
    client = openai.OpenAI(api_key=mltk_config.openai_api_key())
    try:
        completion = client.chat.completions.create(
            model=mltk_config.openai_model(),
            messages=messages,
            functions=functions,
            function_call={"name": "analyze_text"} if functions else None,
            max_tokens=mltk_config.openai_max_tokens()["combined"],
            temperature=mltk_config.openai_temperature(),
        )
        log.info("Successfully retrieved ChatGPT response")
        return completion.choices[0].message
    except openai.OpenAIError:
        log.exception("OpenAI API error")
        raise


def _parse_chatgpt_response(
    message: ChatCompletionMessage,
) -> dict[str, str | list[str] | bool]:
    """Parses the response from ChatGPT.

    Args:
        message: ChatCompletionMessage object.

    Returns:
        Parsed dictionary from JSON response.
    """
    if message.function_call:
        arguments = message.function_call.arguments
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            log.exception(f"Failed to parse function call arguments: {arguments}")
            return {}
    content = message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        log.exception(f"Failed to parse message content: {content}")
        return {}
