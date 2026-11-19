"""
modelo_kmeans.py  ->  MODELO (ML): construir y entrenar el Pipeline de Spark MLlib.

Pipeline = cadena de montaje:
    VectorAssembler (Transformer)  ->  [StandardScaler]  ->  KMeans (Estimator)
Al hacer pipeline.fit(df) se obtiene un PipelineModel que guarda TODOS los pasos,
así se puede aplicar igual a datos nuevos con .transform(df).
"""
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler


def construir_pipeline(columnas, k, seed, escalar=False):
    """Devuelve un Pipeline SIN entrenar: VectorAssembler -> (StandardScaler) -> KMeans."""
    # 1) Junta las columnas (Open, Close) en un único vector por fila: [Open, Close]
    #    escalar=False -> se llama directamente "features"; si se escala, es el paso intermedio
    salida_assembler = "features_sin_escalar" if escalar else "features"
    etapas = [VectorAssembler(inputCols=columnas, outputCol=salida_assembler)]

    # 2) (Opcional) Normaliza cada columna a media 0 y desviación 1.
    #    Con Open y Close no hace falta: las dos están en euros y en la misma escala.
    if escalar:
        etapas.append(StandardScaler(inputCol=salida_assembler, outputCol="features",
                                     withMean=True, withStd=True))

    # 3) KMeans: k = nº de grupos, seed = semilla que fija la inicialización de los centroides
    etapas.append(KMeans(featuresCol="features", predictionCol="prediction", k=k, seed=seed))

    return Pipeline(stages=etapas)


def entrenar(df, columnas, k, seed, escalar=False):
    """Entrena el pipeline completo y devuelve (PipelineModel, predicciones)."""
    pipeline = construir_pipeline(columnas, k, seed, escalar)
    modelo = pipeline.fit(df)                 # ejecuta todas las etapas y entrena KMeans
    predicciones = modelo.transform(df)       # añade las columnas "features" y "prediction"
    return modelo, predicciones


def modelo_kmeans(modelo_pipeline):
    """El KMeansModel es la última etapa del PipelineModel."""
    return modelo_pipeline.stages[-1]


def centros(modelo_pipeline):
    """Centroides de cada cluster como listas de floats (p. ej. [[Open, Close], ...])."""
    return [[float(x) for x in c] for c in modelo_kmeans(modelo_pipeline).clusterCenters()]
