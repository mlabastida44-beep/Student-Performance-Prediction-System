# Submission Draft — Student Performance Prediction System

## 1. SAD mini-analysis
| Element | Response |
|---|---|
| Problem | Academic risk is usually noticed only after poor results. The prototype explores early identification of students likely to fail Mathematics (G3 < 10) using available indicators. |
| User | Faculty member or academic analyst. |
| Inputs | Student attributes (e.g., study time, absences, past failures, G1, G2) uploaded as a CSV. |
| Process | Validate required columns → preprocess (impute, scale, encode) → apply trained Random Forest → classify risk. |
| Outputs | AT RISK / NOT AT RISK label per record, count of at-risk records, downloadable results CSV, model evaluation summary. |
| Limitation | Trained on one Portuguese secondary-school dataset; may not generalize to other institutions or populations. |

## 4. Model evaluation results
**TODO: paste the output of `python train_model.py` (also saved in `evaluation_results.txt`).**

| Metric | Value |
|---|---|
| Accuracy | |
| Precision (At risk) | |
| Recall (At risk) | |
| F1-score (At risk) | |

## 5. Screenshots
TODO: (a) dashboard top; (b) prediction results after uploading `sample_prediction_input.csv`.

## 6. GitHub repository URL
TODO: https://github.com/YOUR-USERNAME/student-performance-prediction

## Test table
| Test | Pass? |
|---|---|
| Run train_model.py | ☐ |
| Run Streamlit app | ☐ |
| Upload sample CSV | ☐ |
| Upload CSV missing a column (delete e.g. `G1`) | ☐ |
| Download results | ☐ |
| Open GitHub repository | ☐ |

## Reflection answers
1. **Target and G3 exclusion:** The target is at-risk status (1 if G3 < 10, else 0). G3 is excluded because the target is computed directly from it; including it would be target leakage, giving near-perfect but meaningless accuracy.
2. **Recall for "At risk":** Of all students who truly ended up at risk, recall is the share the model caught. Low recall means many at-risk students are missed (false negatives), which matters most for a support-flagging tool.
3. **Different institution:** Grading scales, curriculum, demographics, and school context differ from this Portuguese dataset, so feature–outcome relationships may not transfer (dataset shift).
4. **Action after "AT RISK":** Treat it as a prompt to review, not a verdict: check the student's actual records, talk to the student, and offer support such as tutoring or counseling, never penalize based on the prediction alone.
