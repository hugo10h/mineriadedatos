"""
metricas.py  ->  MODELO (evaluación): métricas de calidad de un clustering.

- Silhouette (Ej1): de -1 a 1, cuanto MÁS alta mejor. Mide si cada punto está más
  cerca de los de su grupo (a) que de los del grupo vecino (b): s = (b - a) / max(a, b).
- Davies-Bouldin (Ej2): de 0 a infinito, cuanto MÁS baja mejor. Para cada cluster mira
  su "peor vecino": (dispersión_i + dispersión_j) / distancia_entre_centros_ij.
  Es justo la idea del temario: poca dispersión intracluster y mucha intercluster.
- SSE / inercia (método del codo): suma de distancias^2 de cada punto a su centro.
  Siempre baja al subir k, por eso NO sirve para que Hyperopt elija k; se registra solo
  como información.
"""
import math

from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from modelo_kmeans import centros, modelo_kmeans


def silhouette(predicciones):
    evaluador = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                                    metricName="silhouette",
                                    distanceMeasure="squaredEuclidean")
    return float(evaluador.evaluate(predicciones))


def sse(modelo_pipeline):
    # trainingCost = suma de distancias al cuadrado de cada punto a su centroide
    return float(modelo_kmeans(modelo_pipeline).summary.trainingCost)


def davies_bouldin(modelo_pipeline, predicciones):
    """Índice Davies-Bouldin calculado con Spark (sin UDF de Python)."""
    lista_centros = centros(modelo_pipeline)
    k = len(lista_centros)

    # Los centroides como una columna literal: array de arrays [[c0], [c1], ...]
    centros_col = F.array(*[F.array(*[F.lit(v) for v in c]) for c in lista_centros])

    # 1) Distancia euclídea de cada punto a SU centroide
    df = (predicciones
          .withColumn("vec", vector_to_array("features"))
          .withColumn("centro", F.element_at(centros_col, F.col("prediction") + 1))
          .withColumn("dist", F.sqrt(F.aggregate(
              F.zip_with("vec", "centro", lambda a, b: (a - b) * (a - b)),
              F.lit(0.0), lambda acc, x: acc + x))))

    # 2) Dispersión de cada cluster = distancia media de sus puntos al centro (S_i)
    dispersion = {r["prediction"]: r["S"] for r in
                  df.groupBy("prediction").agg(F.avg("dist").alias("S")).collect()}

    # 3) Para cada cluster i, su peor ratio con otro cluster j; DB = media de esos máximos
    peores = []
    for i in range(k):
        ratios = []
        for j in range(k):
            if i == j:
                continue
            d_centros = math.dist(lista_centros[i], lista_centros[j])
            if d_centros > 0:
                ratios.append((dispersion.get(i, 0.0) + dispersion.get(j, 0.0)) / d_centros)
        peores.append(max(ratios) if ratios else 0.0)
    return float(sum(peores) / k)


def evaluar(modelo_pipeline, predicciones):
    """Todas las métricas juntas en un diccionario (así se registran todas en MLflow)."""
    return {
        "silhouette": silhouette(predicciones),
        "davies_bouldin": davies_bouldin(modelo_pipeline, predicciones),
        "sse": sse(modelo_pipeline),
    }
