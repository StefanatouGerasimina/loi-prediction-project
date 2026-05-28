import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict_loi, load_model


def run_tests(engine, model, df_test):
    """
    Tests for predict loi function using real survey_ids from test set

    Test cases selected from data exploration:
        - Normal survey (random)
        - Survey with fewest questions (8)
        - Survey with most questions (14)
        - Survey with lowest mean LOI (56sec)
        - Survey with highest mean LOI (172sec)
        - Rare category survey (technology, 12 surveys only)
        - Non-existent survey give me ValueError
        - LOI in realistic range
        - Prediction close to actual LOI
    """
    passed = 0
    failed = 0

    def check(condition, msg_pass, msg_fail):
        nonlocal passed, failed
        if condition:
            print(f"passed {msg_pass}")
            passed += 1
        else:
            print(f"didnt pass {msg_fail}")
            failed += 1

    # survey ids
    NORMAL_ID = df_test['survey_id'].iloc[0]
    MIN_Q_ID = '04524663-d921-40ef-99a0-234bcf760e48'  # 8 questions
    MAX_Q_ID = '0923da37-b4b8-426c-a829-f2f3efcefd4a'  # 14 questions
    MIN_LOI_ID = '1d8f1f9b-cb53-4cd1-9a6a-fd75bd9ca388'  # mean LOI=56sec
    MAX_LOI_ID = 'a1fbd060-fa4f-474b-b771-f9b8b7f8f8a0'  # mean LOI=172sec
    TECHNOLOGY_ID = 'ed491772-afe2-4c36-8a47-529c5b27e101'  # rare category
    ART_ID = 'e084ee9b-1b23-4bd5-b1ee-1b2f5ef2b2d6'  # most common

    # test 1: normal survey
    result = predict_loi(NORMAL_ID, engine, model)
    check(
        result['predicted_loi_ms'] > 0,
        f"Test 1 passed: normal survey "
        f"LOI={result['predicted_loi_sec']:.1f}sec",
        "Test 1 failed: non-positive LOI"
    )

    # test 2: survey with fewer questions
    result = predict_loi(MIN_Q_ID, engine, model)
    check(
        result['num_questions'] == 8,
        f"Test 2 passed: min questions survey "
        f"({result['num_questions']} questions, "
        f"LOI={result['predicted_loi_sec']:.1f}sec)",
        f"Test 2 failed: expected 8 questions, "
        f"got {result['num_questions']}"
    )

    # test 3: survey with most questions
    result = predict_loi(MAX_Q_ID, engine, model)
    check(
        result['num_questions'] == 14,
        f"Test 3 passed: max questions survey "
        f"({result['num_questions']} questions, "
        f"LOI={result['predicted_loi_sec']:.1f}sec)",
        f"Test 3 failed: expected 14 questions, "
        f"got {result['num_questions']}"
    )

    # test 4: survey with lowest loi
    result = predict_loi(MIN_LOI_ID, engine, model)
    actual_loi = (df_test[df_test['survey_id'] == MIN_LOI_ID]
                  .groupby('response_id')['answer_time_ms']
                  .sum().mean())
    check(
        result['predicted_loi_ms'] > 0,
        f"Test 4 passed: min LOI survey "
        f"(actual={actual_loi/1000:.1f}sec, "
        f"predicted={result['predicted_loi_sec']:.1f}sec)",
        "Test 4 failed"
    )

    # test 5: survey with highest mean loi
    result = predict_loi(MAX_LOI_ID, engine, model)
    actual_loi = (df_test[df_test['survey_id'] == MAX_LOI_ID]
                  .groupby('response_id')['answer_time_ms']
                  .sum().mean())
    check(
        result['predicted_loi_ms'] > 0,
        f"Test 5 passed: max LOI survey "
        f"(actual={actual_loi/1000:.1f}sec, "
        f"predicted={result['predicted_loi_sec']:.1f}sec)",
        "Test 5 failed"
    )

    # test 6: rare category (technology, 12 surveys) 
    result = predict_loi(TECHNOLOGY_ID, engine, model)
    check(
        result['predicted_loi_ms'] > 0,
        f"Test 6 passed: rare category (technology) "
        f"LOI={result['predicted_loi_sec']:.1f}sec",
        "Test 6 failed"
    )

    # test 7: Most common category (cat_culture and arts)
    result = predict_loi(ART_ID, engine, model)
    check(
        result['predicted_loi_ms'] > 0,
        f"Test 7 passed: common category (culture and arts) "
        f"LOI={result['predicted_loi_sec']:.1f}sec",
        "Test 7 failed"
    )

    # test 8: non existent survey
    try:
        predict_loi(
            '00000000-0000-0000-0000-000000000000',
            engine, model
        )
        check(False, "", "Test 8 failed: should raise ValueError")
    except ValueError:
        check(
            True,
            "Test 8 passed: non-existent survey raises ValueError",
            ""
        )

    # test 9: loi in realistic range 
    # Based on data: min mean loi=56sec, max=172sec
    result = predict_loi(NORMAL_ID, engine, model)
    check(
        30_000 < result['predicted_loi_ms'] < 600_000,
        f"Test 9 passed: LOI in realistic range "
        f"({result['predicted_loi_sec']:.1f}sec)",
        f"Test 9 failed: LOI={result['predicted_loi_sec']:.1f}sec "
        f"out of expected range (30-600sec)"
    )

    # test 10: Pprediction close to actual LOI
    actual = (df_test[df_test['survey_id'] == NORMAL_ID]
              .groupby('response_id')['answer_time_ms']
              .sum().mean())
    result = predict_loi(NORMAL_ID, engine, model)
    error_pct = abs(actual - result['predicted_loi_ms']) / actual * 100
    check(
        error_pct < 50,
        f"Test 10 passed: prediction within 50% of actual "
        f"(error={error_pct:.1f}%, "
        f"actual={actual/1000:.1f}sec, "
        f"predicted={result['predicted_loi_sec']:.1f}sec)",
        f"Test 10 failed: error too large ({error_pct:.1f}%)"
    )

    print(f"Results: {passed} passed, {failed} failed")

if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'results', 'best_lgb_final.pkl')
    data_path = os.path.join(base_dir, 'data', 'data_ready1603.pkl')

    model = load_model(model_path)
    engine = create_engine(
        'postgresql://survey_user:survey_password@localhost:5432/survey_db'
    )

    df = pd.read_pickle(data_path)
    _, test = train_test_split(
        df['response_id'].unique(),
        test_size=0.2,
        random_state=42
    )
    df_test = df[df['response_id'].isin(test)]

    run_tests(engine, model, df_test)