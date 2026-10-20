import glob
import os
import sys

'''
DECLARACIÓN USO DE IA 
Pedido para evitar un fallo del tipo "Python workwd unexpectaly"
'''
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


'''
DECLARACIÓN USO DE IA
Pedido a claude que me ayudase a hacer el spark_session ya que al realizarlo
sin ayuda obtenía varios errores a la hora de ejecutar los codigos
'''
def crear_spark_session(app_name="IBEX35"):
    proyecto_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    builder = (SparkSession.builder
               .appName(app_name)
               .master("local[*]") # Trabajar en mi ordenador usando todos los nucleos
               # --- Delta Lake (obligatorio) ---  Activan Delta
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               # --- Igual que en la P0: evita problemas con nombres de equipo con '_' ---
               .config("spark.driver.host", "localhost")
               .config("spark.driver.bindAddress", "127.0.0.1")
               # Datos pequeños en local: 8 particiones de shuffle en vez de 200
               .config("spark.sql.shuffle.partitions", "8"))

    jars = glob.glob(os.path.join(proyecto_dir, "drivers", "mysql-connector-j*.jar"))
    if jars:
        builder = builder.config("spark.driver.extraClassPath", os.pathsep.join(jars))
    else:
        print("AVISO: no hay mysql-connector-j*.jar en drivers/. "
              "Solo hará falta para publicar la capa gold en MySQL.")

    # Añade los .jar de Delta (la 1ª vez los descarga de Maven: necesita internet)
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
