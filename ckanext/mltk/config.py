from __future__ import annotations

import ckan.plugins.toolkit as tk

NLP_ENGINE = "ckanext.mltk.nlp_engine"

OPENAI_API_KEY = "ckanext.mltk.openai_api_key"
OPENAI_MODEL = "ckanext.mltk.openai_model"
OPENAI_TEMPERATURE = "ckanext.mltk.openai_temperature"
OPENAI_MAX_TOKENS = "ckanext.mltk.openai_max_tokens"

KEYBERT_MODEL_NAME = "ckanext.mltk.keybert_model_name"
KEYBERT_KEYPHRASE_NGRAM_RANGE = "ckanext.mltk.keybert_keyphrase_ngram_range"
KEYBERT_TITLE_NGRAM_RANGE = "ckanext.mltk.keybert_title_ngram_range"
KEYBERT_DESCRIPTION_NGRAM_RANGE = "ckanext.mltk.keybert_description_ngram_range"
KEYBERT_USE_MMR = "ckanext.mltk.keybert_use_mmr"
KEYBERT_DIVERSITY = "ckanext.mltk.keybert_diversity"
KEYBERT_NR_CANDIDATES = "ckanext.mltk.keybert_nr_candidates"
KEYBERT_TOP_N = "ckanext.mltk.keybert_top_n"
KEYBERT_MIN_SCORE = "ckanext.mltk.keybert_min_score"

SENTIMENT_ANALYZER_MODEL = "ckanext.mltk.sentiment_analyzer_model"
SENTIMENT_TEXT_MAX_LENGTH = "ckanext.mltk.sentiment_text_max_length"

NLTK_DOWNLOAD_PUNKT = "ckanext.mltk.nltk_download_punkt"


def nlp_engine():
    """NLP engine to use for NLP tasks (keybert or chatgpt)."""
    return tk.config[NLP_ENGINE]


def openai_api_key():
    """OpenAI API key for ChatGPT."""
    return tk.config[OPENAI_API_KEY]


def openai_model():
    """OpenAI model to use (e.g., 'gpt-3.5-turbo' or 'gpt-4')."""
    return tk.config[OPENAI_MODEL]


def openai_temperature():
    """Temperature parameter for OpenAI API."""
    return tk.config[OPENAI_TEMPERATURE]


def openai_max_tokens():
    """Maximum number of tokens for each part of the generated text."""
    return tk.config[OPENAI_MAX_TOKENS]


def keybert_model_name():
    """KeyBERT model name to use for keyphrase extraction."""
    return tk.config[KEYBERT_MODEL_NAME]


def keybert_keyphrase_ngram_range():
    """N-gram range for keyphrase extraction."""
    return tuple(tk.config[KEYBERT_KEYPHRASE_NGRAM_RANGE])


def keybert_title_ngram_range():
    """N-gram range for title generation."""
    return tuple(tk.config[KEYBERT_TITLE_NGRAM_RANGE])


def keybert_description_ngram_range():
    """N-gram range for description generation."""
    return tuple(tk.config[KEYBERT_DESCRIPTION_NGRAM_RANGE])


def keybert_use_mmr():
    """Use Maximal Marginal Relevance (MMR) for keyphrase extraction."""
    return tk.config[KEYBERT_USE_MMR]


def keybert_diversity():
    """Diversity parameter for keyphrase extraction."""
    return tk.config[KEYBERT_DIVERSITY]


def keybert_nr_candidates():
    """Number of candidates to consider for each part of the generated text."""
    return tk.config[KEYBERT_NR_CANDIDATES]


def keybert_top_n():
    """Number of keyphrases to select for each part of the generated text."""
    return tk.config[KEYBERT_TOP_N]


def keybert_min_score():
    """Minimum score for keyphrase selection."""
    return tk.config[KEYBERT_MIN_SCORE]


def sentiment_analyzer_model():
    """Sentiment analysis model to use."""
    return tk.config[SENTIMENT_ANALYZER_MODEL]


def sentiment_text_max_length():
    """Maximum length of text for sentiment analysis."""
    return tk.config[SENTIMENT_TEXT_MAX_LENGTH]


def nltk_download_punkt():
    """Download 'punkt' tokenizer data for NLTK."""
    return tk.config[NLTK_DOWNLOAD_PUNKT]
