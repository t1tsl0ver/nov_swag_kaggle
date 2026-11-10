# Kaggle Playground Series S6E11 — Predicting Credit Card Fraud

Цель: предсказать вероятность `is_fraud`; метрика — ROC AUC. Соревнование:
https://www.kaggle.com/competitions/playground-series-s6e11/overview

## Структура

- `data/` — локальные `train.csv`, `test.csv`, `sample_submission.csv` (не коммитятся).
- `src/fraud/` — загрузка данных, признаки, обучение и подбор LightGBM.
- `experiments/` — небольшие отчёты о запусках, входят в Git.
- `artifacts/` — OOF, сабмиты и модели, остаются локально.

Данные положи прямо в `data/`. Установка и запуск из корня проекта:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m fraud.cli baseline
.venv\Scripts\python -m fraud.cli tune --trials 8
```

`baseline` использует один фиксированный набор параметров. `tune` сначала
проверяет его, затем случайно выбирает ещё семь наборов из заданной сетки.
Каждый набор оценивается на **одних и тех же пяти фолдах** по общему OOF ROC AUC;
лучший повторно обучается на пяти фолдах для сабмита и анализа признаков.
Подбор таким способом — random search; число попыток меняется через `--trials`.
Все попытки и их оценки записываются в `experiments/<run_id>/trials.csv` сразу
после окончания каждой попытки. Подбор может занять заметно больше времени,
чем `baseline`: по умолчанию это 8 × 5 фолдов для сравнения и ещё 5 для сабмита.

При запуске `experiments/latest_report.txt` перезаписывается. В нём видно
загрузку данных, текущий фолд, число деревьев, ROC AUC на валидации и грубую
оценку оставшегося времени. Лимит в 400 деревьев делает первый прогон быстрым;
ранняя остановка может закончить фолд раньше. Время до лимита — верхняя оценка.

Результат: `experiments/<run_id>/summary.json` с ROC AUC по фолдам и общим OOF,
`permutation_mean.csv` с падением ROC AUC при перемешивании одного признака на
валидации, `feature_importance_mean.csv` с gain/split важностью LightGBM и
`tree_nodes.csv` со всеми узлами разбиения деревьев. `split_gain` показывает
выигрыш функции потерь при построении дерева, а не вклад в ROC AUC. Для решения
об удалении признака смотри сначала `permutation_mean.csv`, затем проверяй
отдельным запуском с теми же фолдами:

```powershell
.venv\Scripts\python -m fraud.cli baseline --drop-feature merchant_category
```

После подбора можно повторять эксперименты с лучшими параметрами:

```powershell
.venv\Scripts\python -m fraud.cli baseline --params-from experiments\RUN_ID\summary.json --drop-feature merchant_category
```

## Проверка слабых признаков после первого baseline

У пяти признаков среднее падение ROC AUC при перемешивании было чуть ниже нуля.
Это слабый сигнал: все они использовались в деревьях, а разница мала по
сравнению с шумом. В наборе `weak_ablation_v1` они исключаются **только из
входа модели**, исходные CSV не меняются. Команда сравнит общий OOF ROC AUC с
первым запуском, сохраняя те же параметры и фолды:

```powershell
.venv\Scripts\python -m fraud.cli baseline --feature-set weak_ablation_v1 --compare-to experiments\20261110_123755_baseline\summary.json
```

Разница появится в `experiments/latest_report.txt` и в `comparison` нового
`summary.json`. Если изменение окажется небольшим, стоит проверить признаки
по одному: совместное удаление не показывает вклад каждого по отдельности.

Отрицательное или нулевое падение ROC AUC при перемешивании может быть шумом
или следствием корреляции признаков. Надёжный критерий — изменение общего OOF
ROC AUC после удаления признака. Большие `oof.csv`, пять файлов моделей и
`submission.csv` лежат в `artifacts/<run_id>/` и не входят в Git. Отчёт по
каждому запуску коммитим отдельно.

