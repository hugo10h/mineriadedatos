"""
datos.py  ->  MODELO (acceso a datos): leer la gold y quedarse con lo que necesita el modelo.
"""
from pyspark.sql import functions as F


def cargar_gold(spark, ruta_fact):
    # La gold de la P2 está exportada en Parquet (una tabla de hechos con todas las empresas)
    return spark.read.parquet(ruta_fact)


def datos_empresa(df_gold, ticker, columnas):
    """Filas de UNA empresa con la clave (Ticker, Fecha) y las columnas pedidas, sin nulos."""
    return (
        df_gold
        .filter(F.col("Ticker") == ticker)              # p. ej. solo BBVA.MC
        .select("Ticker", "Fecha", *columnas)           # clave + features (Open, Close)
        .dropna(subset=columnas)                        # KMeans no admite nulos
        .orderBy("Fecha")                               # orden fijo -> resultado reproducible
        .coalesce(1)                                    # 538 filas: 1 partición basta y fija el orden
    )
