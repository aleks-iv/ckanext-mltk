from __future__ import annotations

import ckan.plugins.toolkit as tk
from ckan.types import ActionResult, Context, DataDict

import ckanext.mltk.config as mltk_config
from ckanext.mltk.utils.chatgpt_nlp import generate_data_with_chatgpt
from ckanext.mltk.utils.common_nlp import (
    get_from_cache,
    perform_sentiment_analysis,
    process_text_resource,
    put_to_cache,
)
from ckanext.mltk.utils.keybert_nlp import (
    extract_keywords_with_keybert,
    generate_description_with_keybert,
    generate_title_with_keybert,
)


@tk.side_effect_free
def generate_resource_keywords(context: Context, data_dict: DataDict) -> ActionResult:
    """Retrieves keywords for the resource."""
    resource_id = data_dict["resource_id"]
    text = process_text_resource(resource_id)

    if mltk_config.nlp_engine() == "chatgpt":
        generated_data = get_from_cache(
            f"mltk:{resource_id}"
        ) or generate_data_with_chatgpt(text)
        put_to_cache(f"mltk:{resource_id}", generated_data)
        keywords = generated_data["keywords"]
    else:
        keywords = extract_keywords_with_keybert(text)

    return {"keywords": keywords}


@tk.side_effect_free
def generate_resource_title(context: Context, data_dict: DataDict) -> ActionResult:
    """Retrieves the title for the resource."""
    resource_id = data_dict.get("resource_id")
    text = process_text_resource(resource_id)

    if mltk_config.nlp_engine() == "chatgpt":
        generated_data = get_from_cache(
            f"mltk:{resource_id}"
        ) or generate_data_with_chatgpt(text)
        put_to_cache(f"mltk:{resource_id}", generated_data)
        title = generated_data["title"]
    else:
        title = generate_title_with_keybert(text)

    return {"title": title}


@tk.side_effect_free
def generate_resource_description(
    context: Context, data_dict: DataDict
) -> ActionResult:
    """Retrieves the description for the resource."""
    resource_id = data_dict.get("resource_id")
    text = process_text_resource(resource_id)

    if mltk_config.nlp_engine() == "chatgpt":
        generated_data = get_from_cache(
            f"mltk:{resource_id}"
        ) or generate_data_with_chatgpt(text)
        put_to_cache(f"mltk:{resource_id}", generated_data)
        description = generated_data["description"]
    else:
        description = generate_description_with_keybert(text)

    return {"description": description}


@tk.side_effect_free
def analyze_resource_sentiment(context: Context, data_dict: DataDict) -> ActionResult:
    """Performs sentiment analysis on the resource content."""
    resource_id = data_dict.get("resource_id")
    text = process_text_resource(resource_id)

    sentiment = perform_sentiment_analysis(text)

    return {"sentiment": sentiment}
