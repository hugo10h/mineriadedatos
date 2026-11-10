from spark_session import crear_spark_session
from exportar_gold import exportar_gold

spark = crear_spark_session()   # CON paréntesis: enciende Spark
exportar_gold(spark)            # exporta las 4 tablas
spark.stop()                    # CON paréntesis: apaga Spark