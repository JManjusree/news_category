# Lab 7 — ML Pipeline Engineering (News Category Classification)

This implementation follows the uploaded churn-project Lab 7 pattern: modular stages, sequential execution, stage-wise validation, saved artifacts, run logs, evaluation, and inference.

## Files to copy
Copy `src/mlops_pipelines/lab7_config.py`, `lab7_components.py`, and `run_lab7_news_pipeline.py` into the same folder in your News Category project.

## Prerequisites
Run from the project root with the project's virtual environment activated:

```powershell
pip install pandas numpy scikit-learn joblib
```

The pipeline expects the existing Lab 3–6 preprocessing output:
- `data/processed/train_33class.csv`
- `data/processed/test_33class.csv`

It also validates the original JSONL file:
- `data/raw/News_Category_Dataset_v3.json`

## Run
From `C:\Users\LENOVO\Desktop\Mlops\assigment`:

```powershell
python src/mlops_pipelines/run_lab7_news_pipeline.py
```

## Pipeline stages
1. Raw dataset validation
2. Run the existing `src/preprocess_pipeline.py` preprocessing stage
3. Processed train/test validation (33 categories and approximately 85/15 split)
4. Train TF-IDF + LinearSVC as one reusable scikit-learn Pipeline
5. Evaluate on the held-out test split
6. Run a five-row inference smoke test
7. Retrain twice and compare model state hashes and predictions

The training configuration preserves the existing News Category project settings:
- TF-IDF: `max_features=50000`, `ngram_range=(1, 2)`, English stop words, `sublinear_tf=True`
- LinearSVC: `C=0.5`, `random_state=42`, `max_iter=5000`

## Generated outputs
- `models/lab7_news_pipeline.joblib`
- `reports/lab7_evaluation_report.json`
- `reports/lab7_pipeline_report.json`
- `artifacts/lab7/inference_predictions.csv`
- `logs/news_category_lab7_pipeline_run_log.json`

## Important
This Lab 7 workflow intentionally does not run the Lab 8 quality gate or Lab 9 failure-injection stages. It uses the existing preprocessed 33-class train/test CSVs; it does not alter the earlier Lab 3–6 scripts or data.
