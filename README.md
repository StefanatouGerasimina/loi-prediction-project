# LOI Prediction 

## Overview
Predicting the **Length of Interview (LOI)** for survey questionnaires.
Given a survey, the model estimates the total time a user will need to complete it.

LOI is estimated by predicting `answer_time_ms` per question,
then summing across all questions in a survey.
This allows us to predict LOI for new surveys with no response history.


---

## References

- Tack et al. (2024). BEA Shared Task on Automated Prediction of
  Item Difficulty and Response Time. ACL Anthology.
- Schneider et al. (2022). Using Attributes of Survey Items to
  Predict Response Times. Field Methods.

---

#### See documentation for more informaiton.
