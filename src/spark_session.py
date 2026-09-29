from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import *
import os
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def crear_spark_session():
    proyecto_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    jdbc_jar_path = os.path.join(
        proyecto_dir,
        "drivers",
        "mssql-jdbc-13.6.0.jre11.jar"
    )

    if not os.path.isfile(jdbc_jar_path):
        raise FileNotFoundError(
            "No se encuentra el driver JDBC de SQL Server. "
            "Debe existir en: " + jdbc_jar_path
        )
    spark_session = (SparkSession.builder
                      .appName("IBEX35")
                      .config("spark.driver.host", "localhost")
                      .config("spark.driver.bindAddress", "127.0.0.1")
                      .config("spark.driver.extraClassPath", jdbc_jar_path)
                      .getOrCreate())
    return spark_session

print("PYTHON USADO:", sys.executable)

'''
DECLARACIÓN USO DE IA

Obtenía errores a la hora de ejecutar el ejercicio, y gracias a
"Claude" lo he solucionado. He realizado así el .conif(...) ya que 
mi nombre de equipo contiene guiones bajos y no son válidos dentro
del host de una URL

Cito : "El nombre de tu equipo es Hugo_Victus_ord, y contiene guiones bajos (_).
Spark usa ese nombre para construir una URL interna de comunicación,
pero los guiones bajos no son válidos en nombres de host dentro de 
una URL — por eso falla al iniciar el SparkContext"

'''