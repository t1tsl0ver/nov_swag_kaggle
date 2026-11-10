# Kaggle Playground Series S6E11 — Predicting Credit Card Fraud

Цель: предсказать вероятность `is_fraud`; метрика — ROC AUC. Соревнование:
https://www.kaggle.com/competitions/playground-series-s6e11/overview

## Структура

- `data/` — локальные `train.csv`, `test.csv`, `sample_submission.csv` (не коммитятся).
- `src/fraud/` — загрузка данных, признаки, обучение и подбор LightGBM.
- `experiments/` — небольшие отчёты о запусках, входят в Git.
- `artifacts/` — OOF, сабмиты и модели, остаются локально.

Данные положи прямо в `data/`. Команды запуска и зависимости появятся вместе с
обучающим кодом. Каждый законченный шаг проекта фиксируем отдельным коммитом.

