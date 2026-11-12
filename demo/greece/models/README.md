# Моделі-переможці

Згенеровано `scripts/07_export_models.py` з останніх запусків. Кожна папка: `card.json` (підхід, правило вибору, метрика на тесті, датасет) і `model.joblib`, якщо модель навчена.
Детектори SD і CD це готові моделі сторонніх авторів, їхні ваги тут не поширюються.

| Код | Задача | Тип | Підхід | Метрика | Тест | Файли |
|---|---|---|---|---|---|---|
| FD | Що згоріло у 2023 році | виміряно | s2_dnbr | F1 | f1 0.872, precision 0.855, recall 0.891 | card.json |
| FM | Які теплові точки це одна пожежа | виміряно | st_2.0km_5d | BCubed F1 | bcubed_f1 0.990 | card.json |
| FG_day_fire_day | Прогноз: день із вогнем | виміряно | lightgbm | PR-AUC | pr_auc 0.257, recall_at_10 0.419 | card.json, model.joblib |
| FG_day_ignition | Прогноз: нове займання | виміряно | lightgbm | PR-AUC | pr_auc 0.278, recall_at_10 0.477 | card.json, model.joblib |
| FG_risk | Які ділянки згорять | виміряно | f22_lgbm_all | ROC-AUC | roc_auc 0.525 | card.json, model.joblib |
| FG_spread | Куди піде вогонь за добу | виміряно | lightgbm_lag | F1 нової площі | f1_24 0.276, f1_48 0.310, front_err_m 1673.417 | card.json |
| LD | Де зрубано ліс | виміряно | lightgbm | F1 | value 0.292 | card.json |
| LG | Де рубатимуть наступного року | виміряно | логрег | частка втрат в 11 % гексів | value 0.513 | card.json |
| AR | Де пожежі повторюються | виміряно | DBSCAN + оболонка | PAI | pai 17.523 | card.json |
| TP | Де ставити вежі | сценарний аналіз | greedy_flat, 300.0 місць | частка пожеж, перша точка | first 0.385 | card.json |
| CP | Де ставити камери | сценарний аналіз | greedy_flat, усі | частка рубок | log 0.783 | card.json |
