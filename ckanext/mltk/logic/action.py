from __future__ import annotations

import logging

import ckan.plugins.toolkit as tk
from ckan.logic import validate
from ckan.types import ActionResult, Context, DataDict

import ckanext.mltk.config as mltk_config
from ckanext.mltk.logic import schema
from ckanext.mltk.utils.chatgpt_nlp import (
    check_if_dataset_description_is_relevant,
    generate_data_with_chatgpt,
)
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

log = logging.getLogger(__name__)


@tk.side_effect_free
@validate(schema.mltk_generate_resource_keywords)
def mltk_generate_resource_keywords(
    context: Context, data_dict: DataDict
) -> ActionResult:
    """Retrieves keywords for the resource.

    Args:
        context: CKAN context object.
        data_dict: Dictionary with 'resource_id' (str) and 'force' (bool).

    Returns:
        Dict with 'keywords' (list of str).

    Raises:
        tk.ObjectNotFound: If resource_id is invalid.
    """
    resource_id = data_dict["resource_id"]
    force = data_dict.get("force", False)
    cache_key = f"mltk:keywords:{resource_id}"

    try:
        text = process_text_resource(resource_id)
        if not text:
            return {"keywords": []}

        if mltk_config.nlp_engine() == "chatgpt":
            if force or not (cached_data := get_from_cache(cache_key)):
                generated_data = generate_data_with_chatgpt(text, entity="resource")
                put_to_cache(cache_key, generated_data)
                keywords = generated_data.get("keywords", [])
            else:
                keywords = cached_data.get("keywords", [])
        else:
            keywords = extract_keywords_with_keybert(text)
    except Exception:
        log.exception(f"Error generating keywords for resource {resource_id}")
        return {"keywords": [], "error": "Failed to generate keywords"}
    else:
        return {"keywords": keywords}


@tk.side_effect_free
@validate(schema.mltk_generate_resource_title)
def mltk_generate_resource_title(context: Context, data_dict: DataDict) -> ActionResult:
    """Retrieves the title for the resource."""
    resource_id = data_dict["resource_id"]
    force = data_dict.get("force", False)
    cache_key = f"mltk:title:{resource_id}"

    try:
        text = process_text_resource(resource_id)
        if not text:
            return {"title": ""}

        if mltk_config.nlp_engine() == "chatgpt":
            if force or not (cached_data := get_from_cache(cache_key)):
                generated_data = generate_data_with_chatgpt(text, entity="resource")
                put_to_cache(cache_key, generated_data)
                title = generated_data.get("title", "")
            else:
                title = cached_data.get("title", "")
        else:
            title = generate_title_with_keybert(text)
    except Exception:
        log.exception(f"Error generating title for resource {resource_id}")
        return {"title": "", "error": "Failed to generate title"}
    else:
        return {"title": title}


@tk.side_effect_free
@validate(schema.mltk_generate_resource_description)
def mltk_generate_resource_description(
    context: Context, data_dict: DataDict
) -> ActionResult:
    """Retrieves the description for the resource."""
    resource_id = data_dict["resource_id"]
    force = data_dict.get("force", False)
    cache_key = f"mltk:description:{resource_id}"

    try:
        text = process_text_resource(resource_id)
        if not text:
            return {"description": ""}

        if mltk_config.nlp_engine() == "chatgpt":
            if force or not (cached_data := get_from_cache(cache_key)):
                generated_data = generate_data_with_chatgpt(text, entity="resource")
                put_to_cache(cache_key, generated_data)
                description = generated_data.get("description", "")
            else:
                description = cached_data.get("description", "")
        else:
            description = generate_description_with_keybert(text)
    except Exception:
        log.exception(f"Error generating description for resource {resource_id}")
        return {"description": "", "error": "Failed to generate description"}
    else:
        return {"description": description}


@tk.side_effect_free
@validate(schema.mltk_generate_dataset_description)
def mltk_generate_dataset_description(
    context: Context, data_dict: DataDict
) -> ActionResult:
    """Generates a description for the dataset based on its resources content."""
    dataset_id = data_dict["id"]
    force = data_dict.get("force", False)
    cache_key = f"mltk:dataset_description:{dataset_id}"

    try:
        resources = tk.get_action("package_show")({}, {"id": dataset_id})["resources"]
        if not resources:
            return {"description": "", "report": "No resources found"}

        resource_texts = [process_text_resource(res["id"]) for res in resources]
        text = " ".join(resource_texts)[:4096]  # Truncate to 4096 chars as an example

        if mltk_config.nlp_engine() == "chatgpt":
            if force or not (cached_data := get_from_cache(cache_key)):
                generated_data = generate_data_with_chatgpt(text, entity="dataset")
                put_to_cache(cache_key, generated_data)
                description = generated_data.get("description", "")
            else:
                description = cached_data.get("description", "")
        else:
            description = generate_description_with_keybert(text)
    except Exception:
        log.exception(f"Error generating dataset description for {dataset_id}")
        return {"description": "", "error": "Failed to generate dataset description"}
    else:
        return {"description": description}


@tk.side_effect_free
@validate(schema.mltk_validate_and_improve_dataset_description)
def mltk_validate_and_improve_dataset_description(
    context: Context, data_dict: DataDict
) -> ActionResult:
    """Validates the dataset description and generates a new one if necessary."""
    dataset_id = data_dict["id"]
    try:
        dataset = tk.get_action("package_show")({}, {"id": dataset_id})
        resources = dataset["resources"]
        if not resources:
            return {
                "is_relevant": False,
                "report": "No resources found",
                "description": None,
            }

        resource_texts = [process_text_resource(res["id"]) for res in resources]
        resource_texts = " ".join(resource_texts)[:4096]  # Truncate for safety

        if not dataset.get("notes"):
            try:
                description = tk.get_action("mltk_generate_dataset_description")(
                    context, data_dict
                )["description"]

            except Exception:
                log.exception(f"Failed to generate description for {dataset_id}")
                return {
                    "is_relevant": False,
                    "report": "No description found. Could not generate a new one.",
                    "description": None,
                }
            else:
                return {
                    "is_relevant": False,
                    "report": "No description found. Generated a new one.",
                    "description": description,
                }

        if mltk_config.nlp_engine() == "chatgpt":
            is_relevant = check_if_dataset_description_is_relevant(
                dataset["notes"], resource_texts
            )
            if is_relevant:
                return {
                    "is_relevant": True,
                    "report": "Description is relevant to resources.",
                    "description": None,
                }
            description = tk.get_action("mltk_generate_dataset_description")(
                context, data_dict
            )["description"]
            return {
                "is_relevant": False,
                "report": "Description is not relevant. Generated a new one.",
                "description": description,
            }
    except Exception:
        log.exception(f"Error validating description for {dataset_id}")
        return {
            "is_relevant": False,
            "report": "Validation failed due to an error.",
            "description": None,
        }


@tk.side_effect_free
@validate(schema.mltk_analyze_resource_sentiment)
def mltk_analyze_resource_sentiment(
    context: Context, data_dict: DataDict
) -> ActionResult:
    """Performs sentiment analysis on the resource content."""
    resource_id = data_dict["resource_id"]
    cache_key = f"mltk:sentiment:{resource_id}"

    try:
        text = process_text_resource(resource_id)
    except Exception:
        log.exception(f"Error analyzing sentiment for resource {resource_id}")
        return {"sentiment": "neutral", "error": "Failed to analyze sentiment"}
    else:
        if not text:
            return {"sentiment": "neutral"}

        if cached_data := get_from_cache(cache_key):
            sentiment = cached_data
        else:
            sentiment = perform_sentiment_analysis(text)
            put_to_cache(cache_key, sentiment)

        return {"sentiment": sentiment}
