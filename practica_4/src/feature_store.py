"""
feature_store.py  ->  MODELO: feature store "hecho a mano" sobre una tabla Delta.

Tabla: clave (Ticker, Fecha) + columnas de features + columna version.
Para entrenar se LEE una versión concreta en lugar de recalcular las features.
"""
from pyspark.sql import functions as F


def guardar_features(df_features, version, ruta):
    """Guarda (o sustituye) una versión de las features.

    replaceWhere hace que, si se vuelve a ejecutar, la versión 1 se reemplace en vez de
    duplicarse (escritura idempotente). Las demás versiones no se tocan.
    """
    (df_features
     .withColumn("version", F.lit(version))
     .write.format("delta")
     .mode("overwrite")
     .option("replaceWhere", f"version = {version}")
     .save(ruta))


def leer_features(spark, ruta, version, ticker=None):
    """Recupera una versión concreta (y opcionalmente solo una empresa)."""
    df = spark.read.format("delta").load(ruta).filter(F.col("version") == version)
    if ticker is not None:
        df = df.filter(F.col("Ticker") == ticker)
    # Mismo orden y particionado que datos.datos_empresa -> mismo resultado de KMeans
    return df.drop("version").orderBy("Fecha").coalesce(1)


def versiones_disponibles(spark, ruta):
    return [r["version"] for r in
            spark.read.format("delta").load(ruta).select("version").distinct()
            .orderBy("version").collect()]
