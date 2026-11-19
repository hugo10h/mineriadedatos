"""
busqueda_hyperopt.py  ->  MODELO: búsqueda automática del mejor k con Hyperopt.

Hyperopt siempre MINIMIZA el "loss". Como la métrica elegida es Davies-Bouldin
(cuanto más baja, mejor), se devuelve tal cual, sin cambiarle el signo.
(Con silhouette habría que devolver -silhouette).
"""

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from entrenamiento import entrenar_evaluar_registrar


def buscar_mejor_k(df, ticker, columnas, k_min, k_max, max_evals,
                   seed_modelo, seed_hyperopt, escalar=False):
    # Espacio de búsqueda: enteros de k_min a k_max (quniform devuelve floats: 2.0, 3.0...)
    espacio = {"k": hp.quniform("k", k_min, k_max, 1)}

    df = df.cache()          # se entrena muchas veces con los mismos datos: se guardan en memoria

    def objetivo(params):
        k = int(params["k"])
        # Cada intento de Hyperopt = una run de MLflow (dentro se abre mlflow.start_run)
        res = entrenar_evaluar_registrar(df, ticker, columnas, k, seed_modelo, escalar)
        print(f"   probado k={k:>2} | davies_bouldin={res['davies_bouldin']:.4f} "
              f"| silhouette={res['silhouette']:.4f}")
        return {"loss": res["davies_bouldin"], "status": STATUS_OK}

    trials = Trials()        # historial de todos los intentos (params + loss)
    mejor = fmin(
        fn=objetivo,
        space=espacio,
        algo=tpe.suggest,                                  # búsqueda bayesiana TPE
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed_hyperopt),       # 2ª semilla: fija QUÉ k se prueban
        show_progressbar=False,
    )
    return int(mejor["k"]), trials
