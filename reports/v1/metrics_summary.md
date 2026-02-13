# V1 Baselines - Metrics Summary

## IN

| Model | MAE | RMSE | SMAPE |
|---|---:|---:|---:|
| `catboost` | 1.257309 | 3.024605 | 12.681425 |
| `catboost_log` | 1.223729 | 3.156499 | 12.107335 |
| `lightgbm` | 1.242295 | 2.880024 | 13.329926 |
| `lightgbm_log` | 1.194402 | 2.868304 | 12.309949 |

- Best MAE: `lightgbm_log` = 1.194402
- Best RMSE: `lightgbm_log` = 2.868304
- Best SMAPE: `catboost_log` = 12.107335

## US

| Model | MAE | RMSE | SMAPE |
|---|---:|---:|---:|
| `catboost` | 19849.060243 | 134585.404314 | 28.863805 |
| `catboost_log` | 18129.817629 | 136830.476882 | 24.799711 |
| `lightgbm` | 19479.752629 | 132525.849255 | 26.699462 |
| `lightgbm_log` | 16617.875371 | 134213.112215 | 20.994136 |

- Best MAE: `lightgbm_log` = 16617.875371
- Best RMSE: `lightgbm` = 132525.849255
- Best SMAPE: `lightgbm_log` = 20.994136
