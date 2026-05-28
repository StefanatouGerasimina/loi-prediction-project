import numpy as np
import pandas as pd
import textstat

from constants import (
    BROAD_CAT,
    MANUAL_CAT_MAPPING, 
    FEATURE_COLS,
    DIFFICULTY_WEIGHTS, 
    MOST_COMMON_CAT
)


def _safe_prompt_feature(prompt, func):
    """
    Returns 0 if prompt is null or empty.
    """
    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        return 0
    try:
        return func(prompt)
    except Exception:
        return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles edge cases:
    - Null/empty prompts sets features default to 0
    - Unknown category sets fallback to most common category
    - Null avg_answer_choice_length sets 0 (open_ended/rating
      have no predefined choices)

    Input df must have columns:
        question_type, prompt, order,
        num_answer_choices, avg_answer_choice_length, category
    """

    df = df.copy()
    n_questions = len(df)

    # handle null prompts 
    null_prompts = df['prompt'].isna().sum()
    if null_prompts > 0:
        print(f"{null_prompts} null prompts are filling with empty string")
        df['prompt'] = df['prompt'].fillna('')

    # question type encoding
    for qt in ['open_ended', 'multiple_selection','single_selection', 'rating']:
        df[f'qt_{qt}'] = (df['question_type'] == qt).astype(int)

    # prompt features
    df['word_count'] = df['prompt'].apply(
        lambda x: _safe_prompt_feature(x, lambda p: len(p.split())))
    df['prompt_length'] = df['prompt'].apply(
        lambda x: _safe_prompt_feature(x, len))
    df['avg_word_length'] = df['prompt'].apply(
        lambda x: _safe_prompt_feature(x, lambda p: np.mean([len(w) for w in p.split()])))
    df['pct_big_words'] = df['prompt'].apply(
        lambda x: _safe_prompt_feature(
            x, lambda p: sum(len(w) >= 7 for w in p.split()) / len(p.split())))
    df['type_token_ratio'] = df['prompt'].apply(
        lambda x: _safe_prompt_feature(
            x, lambda p: len(set(p.lower().split())) / len(p.split())))
    df['readability'] = df['prompt'].apply(
        lambda x: _safe_prompt_feature(x, textstat.flesch_reading_ease))
    df['has_question_mark'] = df['prompt'].str.endswith('?').astype(int)
    df['is_personal'] = df['prompt'].str.lower().str.contains(
        r'\byou\b|\byour\b', na=False).astype(int)
    df['has_analytical'] = df['prompt'].str.lower().str.contains(
        r'\bif\b|\btherefore\b|\bbecause\b|\bcompare\b|\banalyze\b|\bassess\b',
        na=False).astype(int)
    df['has_negation'] = df['prompt'].str.lower().str.contains(
        r'\bnot\b|\bnever\b|\bno\b|\bnone\b', na=False).astype(int)
    df['has_temporal'] = df['prompt'].str.lower().str.contains(
        r'\busually\b|\btypically\b|\boften\b|\bsometimes\b',
        na=False).astype(int)
    df['has_numbers'] = df['prompt'].str.contains(
        r'\d', na=False).astype(int)
    df['is_complex_prompt'] = df['prompt'].str.lower().str.contains(
        r'\bwhy\b|\bhow\b|\bdescribe\b|\bexplain\b',
        na=False).astype(int)
    df['starts_with_why'] = df['prompt'].str.lower().str.startswith('why').astype(int)
    df['starts_with_how'] = df['prompt'].str.lower().str.startswith('how').astype(int)
    df['starts_with_describe'] = df['prompt'].str.lower().str.startswith('describe').astype(int)
    df['starts_with_rate'] = df['prompt'].str.lower().str.startswith('rate').astype(int)

    # answer choice features 
    # NULL for open_ended/rating set to 0 (no predefined choices)
    df['avg_answer_choice_length'] = df['avg_answer_choice_length'].fillna(0)

    # survey feats
    df['num_questions_in_survey'] = n_questions
    df['position_in_survey'] = df['order'] / n_questions

    for qt in ['open_ended', 'multiple_selection',
               'single_selection', 'rating']:
        df[f'num_{qt}_questions'] = df[f'qt_{qt}'].sum()

    df['open_ended_ratio'] = (
        df['num_open_ended_questions'] / n_questions
    )

    # weighted diffivulty per number of questions based on the median of answer time for each type
    df['survey_difficulty_score'] = (
        df['num_open_ended_questions'] * DIFFICULTY_WEIGHTS['open_ended'] +
        df['num_multiple_selection_questions'] * DIFFICULTY_WEIGHTS['multiple_selection'] +
        df['num_single_selection_questions']   * DIFFICULTY_WEIGHTS['single_selection'] +
        df['num_rating_questions'] * DIFFICULTY_WEIGHTS['rating']
    ) / n_questions

    # one hot encoding
    category_clean = (
        df['category'].iloc[0]
        .lower().strip().replace('-', ' ')
    )
    category_broad = MANUAL_CAT_MAPPING.get(category_clean, None)
    
    print(f"Category clean: {category_clean}")
    print(f"Category broad: {category_broad}")
    print(f"Cat columns created: {[f'cat_{c}' for c in BROAD_CAT]}")

    if category_broad is None:
        print(f"Unknown category '{category_clean}'. fallback to {MOST_COMMON_CAT}")
        category_broad = MOST_COMMON_CAT

    for cat in BROAD_CAT:
        df[f'cat_{cat}'] = int(category_broad == cat)

    return df