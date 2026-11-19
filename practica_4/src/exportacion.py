"""
exportacion.py  ->  MODELO: deja el resultado del clustering en Parquet para Power BI (Clase 3).
"""
from pyspark.sql import functions as F


def exportar_clusters(lista_predicciones, ruta):
    """Une las predicciones de varias empresas y las guarda en un único Parquet."""
    df = None
    for pred in lista_predicciones:
        parte = pred.select("Ticker", "Fecha", "Open", "Close",
                            F.col("prediction").alias("Cluster"))
        df = parte if df is None else df.unionByName(parte)
    df.coalesce(1).write.mode("overwrite").parquet(ruta)
    return df
