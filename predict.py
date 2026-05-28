import pickle
import warnings
import pandas as pd
from constants import FEATURE_COLS
from features  import engineer_features


def load_model(model_path: str):
    """load pickle model lightgbm"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_loi(survey_id: str, engine, model) -> dict:
    """
    Predicts the LOI for a given survey

    Attributes:
        survey_id (str): UUID of the survey to predict
        engine (sqlalchemy engine): active database connection
        model: trained lightgbm model

    Returns:
        dict:
            survey_id : str
            predicted_loi_ms (float): total predicted time in ms
            predicted_loi_sec (float): total predicted time in sec
            predicted_loi_min (float): total predicted time in min
            num_questions(int): number of questions in survey
            per_question (df): per-question breakdown
    """

    df = pd.read_sql(f"""
        SELECT
            q.question_id,
            q.question_type,
            q.prompt,
            q."order",
            q.max_rating,
            q.survey_id,
            s.category,
            COUNT(a.answer_id) as num_answer_choices,
            AVG(LENGTH(a.answer_value)) as avg_answer_choice_length
        FROM question q
        JOIN survey s ON q.survey_id = s.id
        LEFT JOIN answer a ON q.question_id = a.question_id
        WHERE q.survey_id = '{survey_id}'
        GROUP BY
            q.question_id, q.question_type, q.prompt,
            q."order", q.max_rating, q.survey_id, s.category
        ORDER BY q."order"
    """, engine)

    # validation of input
    if len(df) == 0:
        raise ValueError(
            f"survey_id: {survey_id} not found in database"
        )
    if len(df) < 2:
        raise ValueError(
            f"survey_id: {survey_id} has {len(df)} question"
            f" minimum of 2 required"
        )

    # features 
    df = engineer_features(df)
    
    print("Columns after engineer_features:")
    print([c for c in df.columns if c.startswith('cat_')])

    # null checking 
    nulls = df[FEATURE_COLS].isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print(f"filling nulls: {nulls.to_dict()}")
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    # prediction
    X = df[FEATURE_COLS].astype(float)
    df['pred_ms'] = model.predict(X)

    # output validation
    if (df['pred_ms'] < 0).any():
        raise ValueError( # sanity check for negative predictions
            f"Negative predictions detected: "
            f"{df['pred_ms'].min():.0f}ms"
        )

    loi_ms = df['pred_ms'].sum()

    if loi_ms > 3_600_000:  # loi > 1 hour
        print(f"unusually high for LOI: record {loi_ms/60000:.1f} min")

    return {
        'survey_id': survey_id,
        'predicted_loi_ms': loi_ms,
        'predicted_loi_sec': loi_ms / 1000,
        'predicted_loi_min': loi_ms / 60000,
        'num_questions': len(df),
        'per_question': df[['question_id', 'question_type', 'prompt', 'pred_ms']]
    }
