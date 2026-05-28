-- init.sql
-- This script runs automatically when the container starts for the first time

-- Create survey table
CREATE TABLE IF NOT EXISTS survey (
    id UUID PRIMARY KEY,
    topic TEXT NOT NULL,
    category TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create question table
CREATE TABLE IF NOT EXISTS question (
    question_id UUID PRIMARY KEY,
    survey_id UUID NOT NULL,
    question_type TEXT NOT NULL,
    prompt TEXT NOT NULL,
    "order" INTEGER NOT NULL,
    max_rating INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_survey FOREIGN KEY (survey_id) REFERENCES survey(id) ON DELETE CASCADE,
    CONSTRAINT unique_survey_order UNIQUE(survey_id, "order")
);

-- Create answer table (for question options)
CREATE TABLE IF NOT EXISTS answer (
    answer_id UUID PRIMARY KEY,
    question_id UUID NOT NULL,
    answer_order INTEGER NOT NULL,
    answer_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_question FOREIGN KEY (question_id) REFERENCES question(question_id) ON DELETE CASCADE,
    CONSTRAINT unique_question_answer_order UNIQUE(question_id, answer_order)
);

-- Create survey_response table
create table if not exists survey_response (
    id serial PRIMARY KEY,
    response_id UUID NOT NULL,
    question_id UUID NOT NULL,
    answer_time_ms INTEGER NOT NULL,
    CONSTRAINT fk_survey_response_question FOREIGN KEY (question_id) REFERENCES question(question_id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_question_survey_id ON question(survey_id);
CREATE INDEX IF NOT EXISTS idx_answer_question_id ON answer(question_id);
CREATE INDEX IF NOT EXISTS idx_survey_category ON survey(category);
CREATE INDEX IF NOT EXISTS idx_question_type ON question(question_type);

-- Grant privileges
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO survey_user;