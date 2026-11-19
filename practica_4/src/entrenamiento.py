"""
entrenamiento.py  ->  MODELO: une las piezas en un único paso reutilizable:
    entrenar el pipeline  ->  calcular métricas  ->  registrar la run en MLflow.
Lo usan el Ej1 (bucle manual), el Ej2/Ej5 (función objetivo de Hyperopt) y el Ej4.
"""
from metricas import evaluar
from modelo_kmeans import entrenar
from seguimiento_mlflow import registrar_run


def entrenar_evaluar_registrar(df, ticker, columnas, k, seed,
                               escalar=False, guardar_modelo=False, origen="gold"):
    modelo, predicciones = entrenar(df, columnas, k, seed, escalar)
    metricas = evaluar(modelo, predicciones)

    parametros = {
        "algoritmo": "KMeans",
        "k": k,
        "seed": seed,
        "ticker": ticker,
        "features": ",".join(columnas),
        "escalado": escalar,
        "origen_datos": origen,          # "gold" o "feature_store_v1" (trazabilidad)
    }
    run_id = registrar_run(f"k={k}", parametros, metricas,
                           modelo if guardar_modelo else None)

    return {"k": k, "run_id": run_id, "modelo": modelo,
            "predicciones": predicciones, **metricas}
