from __future__ import annotations

from ckan import types
from ckan.logic.schema import validator_args


@validator_args
def mltk_generate_resource_keywords(
    not_empty: types.Validator,
    boolean_validator: types.Validator,
    default: types.Validator,
) -> types.Schema:
    return {"resource_id": [not_empty], "force": [boolean_validator, default(False)]}


@validator_args
def mltk_generate_resource_title(
    not_empty: types.Validator,
    boolean_validator: types.Validator,
    default: types.Validator,
) -> types.Schema:
    return {"resource_id": [not_empty], "force": [boolean_validator, default(False)]}


@validator_args
def mltk_generate_resource_description(
    not_empty: types.Validator,
    boolean_validator: types.Validator,
    default: types.Validator,
) -> types.Schema:
    return {"resource_id": [not_empty], "force": [boolean_validator, default(False)]}


@validator_args
def mltk_generate_dataset_description(
    not_empty: types.Validator,
    boolean_validator: types.Validator,
    default: types.Validator,
) -> types.Schema:
    return {"id": [not_empty], "force": [boolean_validator, default(False)]}


@validator_args
def mltk_validate_and_improve_dataset_description(
    not_empty: types.Validator,
    boolean_validator: types.Validator,
    default: types.Validator,
) -> types.Schema:
    return {"id": [not_empty], "force": [boolean_validator, default(False)]}


@validator_args
def mltk_analyze_resource_sentiment(
    not_empty: types.Validator,
    boolean_validator: types.Validator,
    default: types.Validator,
) -> types.Schema:
    return {"resource_id": [not_empty]}
