from __future__ import annotations

import logging

import nltk
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

import ckanext.mltk.config as mltk_config

log = logging.getLogger(__name__)

kw_model: KeyBERT | None = None


def _init_kw_model():
    """Initializes the KeyBERT model with the specified parameters."""
    global kw_model  # noqa
    if kw_model is not None:
        return
    embedding_model = SentenceTransformer(mltk_config.keybert_model_name())
    kw_model = KeyBERT(model=embedding_model)


def extract_keywords_with_keybert(text: str) -> list:
    """Extracts keywords using KeyBERT."""
    _init_kw_model()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=mltk_config.keybert_keyphrase_ngram_range(),
        stop_words="english",
        use_mmr=mltk_config.keybert_use_mmr(),
        diversity=mltk_config.keybert_diversity(),
        nr_candidates=mltk_config.keybert_nr_candidates()["keywords"],
        top_n=mltk_config.keybert_top_n()["keywords"],
    )

    return [
        (kw, round(score, 4))
        for kw, score in keywords
        if score > mltk_config.keybert_min_score()
    ]


def generate_title_with_keybert(text: str) -> str:
    """Generates a title using KeyBERT-extracted keyphrases."""
    _init_kw_model()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=mltk_config.keybert_title_ngram_range(),
        stop_words="english",
        use_mmr=mltk_config.keybert_use_mmr(),
        diversity=mltk_config.keybert_diversity(),
        nr_candidates=mltk_config.keybert_nr_candidates()["title"],
        top_n=mltk_config.keybert_top_n()["title"],
    )

    if keywords:
        keyphrases = [kw[0].capitalize() for kw, _ in keywords]
        title = " - ".join(keyphrases)
    else:
        title = "Untitled"

    return title


def generate_description_with_keybert(text: str) -> str:
    """Generates a description using KeyBERT-extracted keyphrases."""
    _init_kw_model()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=mltk_config.keybert_description_ngram_range(),
        stop_words="english",
        use_mmr=mltk_config.keybert_use_mmr(),
        diversity=mltk_config.keybert_diversity(),
        nr_candidates=mltk_config.keybert_nr_candidates()["description"],
        top_n=mltk_config.keybert_top_n()["description"],
    )

    if keywords:
        keyphrases = [kw[0] for kw, _ in keywords]
        description = _construct_description_from_keyphrases(text, keyphrases)
    else:
        description = "No description available."

    return description


def _construct_description_from_keyphrases(text: str, keyphrases: list) -> str:
    """Constructs a description.

    The description is constructed by selecting the first three sentences that contain
    at least one of the keyphrases. If no such sentences are found, a generic
    description is generated using the keyphrases themselves.
    """
    _init_kw_model()
    sentences = nltk.sent_tokenize(text)
    description_sentences = []

    for sentence in sentences:
        for keyphrase in keyphrases:
            if keyphrase.lower() in sentence.lower():
                description_sentences.append(sentence.strip())
                break

    if description_sentences:
        description = " ".join(description_sentences[:3])
    else:
        description = (
            "This resource covers topics such as "
            + ", ".join(keyphrases[:-1])
            + ", and "
            + keyphrases[-1]
            + "."
        )

    return description
