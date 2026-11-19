"""
seguimiento_mlflow.py  ->  MODELO (MLOps): todo lo que habla con MLflow.

Jerarquía:  experimento (p. ej. "kmeans_BBVA")  ->  muchas runs (una por cada k probado)
Cada run guarda: parámetros (k, seed, columnas...), métricas y, si se pide, el modelo.
"""
import os

# Las versiones nuevas de MLflow (3.x) bloquean el almacenamiento en carpeta (file:///mlruns)
# y recomiendan una base de datos. El enunciado usa la carpeta local, así que se permite
# explícitamente. Tiene que estar ANTES de crear el cliente de MLflow.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.spark
import pyspark


def configurar_mlflow(carpeta_mlruns, nombre_experimento):
    """Indica DÓNDE se guarda todo (carpeta local mlruns) y QUÉ experimento está activo."""
    carpeta_mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(carpeta_mlruns.as_uri())     # file:///C:/.../practica_4/mlruns
    experimento = mlflow.set_experiment(nombre_experimento)  # lo crea si no existe
    return experimento.experiment_id


def registrar_run(nombre_run, parametros, metricas, modelo=None):
    """Abre una run, guarda parámetros y métricas (y el modelo si se pasa) y devuelve su run_id."""
    with mlflow.start_run(run_name=nombre_run) as run:   # el with cierra la run aunque haya error
        mlflow.log_params(parametros)                     # varios log_param de golpe
        mlflow.log_metrics(metricas)                      # varios log_metric de golpe
        if modelo is not None:
            # Guarda el PipelineModel como artefacto de la run (se recupera con mlflow.spark.load_model)
            # (2º argumento posicional = carpeta del artefacto; vale en MLflow 2 y 3)
            # pip_requirements: se indica la versión de PySpark a mano. Si no, MLflow intenta
            # adivinarla cargando el modelo en otro proceso (tarda ~1 min y en Windows falla).
            mlflow.spark.log_model(modelo, "modelo",
                                   pip_requirements=[f"pyspark=={pyspark.__version__}"])
        return run.info.run_id


def obtener_runs_ordenadas(nombre_experimento, metrica, ascendente):
    """Ej3: todas las runs de un experimento, ordenadas por una métrica (DataFrame de pandas)."""
    experimento = mlflow.get_experiment_by_name(nombre_experimento)
    if experimento is None:
        raise ValueError(f"No existe el experimento '{nombre_experimento}'")
    runs = mlflow.search_runs(experiment_ids=[experimento.experiment_id])
    columna = f"metrics.{metrica}"
    runs = runs.dropna(subset=[columna])                  # quita runs sin esa métrica
    return runs.sort_values(by=columna, ascending=ascendente).reset_index(drop=True)


def listar_experimentos():
    """Ej5: experimentos disponibles (list_experiments ya no existe en MLflow 2/3)."""
    return [e for e in mlflow.search_experiments() if e.name != "Default"]


def mejor_run(nombre_experimento, metrica, ascendente):
    """Ej5: la primera run tras ordenar = la mejor según la métrica."""
    runs = obtener_runs_ordenadas(nombre_experimento, metrica, ascendente)
    return runs.iloc[0] if len(runs) else None
