# V1 Baselines - Metrics Summary

## IN

| Model | MAE | RMSE | SMAPE |
|---|---:|---:|---:|
| `catboost` | 1.345397 | 3.249023 | 13.184792 |
| `catboost_log` | 1.327805 | 3.356626 | 12.874680 |
| `lightgbm` | 1.568325 | 3.246391 | 17.464329 |
| `lightgbm_log` | 1.512532 | 3.322357 | 15.989162 |

- Best MAE: `catboost_log` = 1.327805
- Best RMSE: `lightgbm` = 3.246391
- Best SMAPE: `catboost_log` = 12.874680

## US

| Model | MAE | RMSE | SMAPE |
|---|---:|---:|---:|
| `catboost` | 18796.517123 | 135974.784331 | 26.872156 |
| `catboost_log` | 18805.566559 | 137446.589298 | 25.568284 |
| `lightgbm` | 27421.864206 | 137126.400130 | 48.719693 |
| `lightgbm_log` | 26866.600502 | 142544.739209 | 43.871030 |

- Best MAE: `catboost` = 18796.517123
- Best RMSE: `catboost` = 135974.784331
- Best SMAPE: `catboost_log` = 25.568284
