from __future__ import annotations

import json
import logging

import nltk
import pdftotext
from transformers import pipeline

import ckan.plugins.toolkit as tk
from ckan.lib.redis import connect_to_redis
from ckan.lib.uploader import get_resource_uploader

import ckanext.mltk.config as mltk_config

logger = logging.getLogger(__name__)

CACHE_EXPIRATION = 24 * 60 * 60  # 24 hours

sentiment_analyzer: pipeline | None = None


def _init_sentiment_analyzer():
    """Initializes the sentiment analysis pipeline."""
    global sentiment_analyzer  # noqa
    if sentiment_analyzer is not None:
        return
    if mltk_config.nltk_download_punkt():
        nltk.download("punkt")
    sentiment_analyzer = pipeline(mltk_config.sentiment_analyzer_model())


def process_text_resource(resource_id: str) -> str:
    """Processes the resource to extract text content."""
    res_dict = _get_resource_dict(resource_id)
    file_path = get_resource_uploader(res_dict).get_path(res_dict["id"])
    file_type = res_dict["format"].lower()
    text = _get_file_content(file_path, file_type)

    if not text:
        raise tk.ValidationError(
            {"resource_id": ["No text extracted from the resource."]}
        )

    return text


def _get_resource_dict(resource_id: str) -> dict:
    """Fetches and returns the resource dictionary given a resource ID."""
    try:
        return tk.get_action("resource_show")({}, {"id": resource_id})
    except tk.ObjectNotFound:
        logger.exception("Resource not found")
        raise tk.ObjectNotFound from None


def _get_file_content(file_path: str, file_type: str) -> str:
    """Extracts content from a given file based on its type."""
    try:
        if file_type == "pdf":
            return _extract_text_from_pdf(file_path)
        elif file_type == "txt":  # noqa
            return _extract_text_from_txt(file_path)
        else:
            logger.error("Unsupported file type %s", file_type)

    except Exception as e:
        logger.exception("Error reading file %s", file_path)
        raise tk.ValidationError({"file": ["Error reading file."]}) from e


def _extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    with open(file_path, "rb") as file:
        return "\n".join(list(pdftotext.PDF(file)))


def _extract_text_from_txt(file_path: str) -> str:
    """Extracts text from a TXT file."""
    with open(file_path, encoding="utf-8", errors="ignore") as file:
        return file.read()


def get_from_cache(cache_key: str) -> dict:
    """Retrieves cached data from Redis."""
    redis = connect_to_redis()
    try:
        cached_json = redis.get(cache_key)
        if cached_json:
            return json.loads(cached_json)
    except Exception as e:
        logger.exception("Redis get error")
        raise tk.ValidationError({"cache": ["Error retrieving cached data."]}) from e
    return {}


def put_to_cache(cache_key: str, data: dict):
    """Caches data in Redis."""
    redis = connect_to_redis()
    redis.set(cache_key, json.dumps(data), CACHE_EXPIRATION)


def perform_sentiment_analysis(text: str) -> list:
    """Performs sentiment analysis on the given text."""
    _init_sentiment_analyzer()
    return sentiment_analyzer(
        text[: mltk_config.sentiment_text_max_length()], truncation=True
    )
