from config import TABLAS_GOLD, ruta_export


def exportar_tabla(spark, nombre):
    # Lee la tabla gold en formato Delta
    df = spark.read.format("delta").load(TABLAS_GOLD[nombre])

    # Muestra cuántas filas y qué columnas tiene (lo necesitaremos para Power BI)
    print(f"--- {nombre}: {df.count()} filas ---")
    df.printSchema()

    # coalesce(1) = un solo fichero; overwrite = reemplaza si ya existía
    df.coalesce(1).write.mode("overwrite").parquet(ruta_export(nombre))


def exportar_gold(spark):
    # Repite lo mismo para cada tabla del diccionario (las 4)
    for nombre in TABLAS_GOLD:
        exportar_tabla(spark, nombre)