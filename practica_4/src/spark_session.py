import os
import sys

'''
DECLARACIÓN USO DE IA
Pedido para evitar un fallo del tipo "Python worker exited unexpectedly"
'''
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Ruta de la carpeta del proyecto (practica_4): sube dos niveles desde este fichero
_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HADOOP_HOME debe apuntar a la carpeta que CONTIENE a "bin" (no a bin)
os.environ["HADOOP_HOME"] = _RAIZ

# Añade bin al PATH para que Windows encuentre winutils.exe y hadoop.dll
os.environ["PATH"] = os.path.join(_RAIZ, "bin") + os.pathsep + os.environ["PATH"]


from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


'''
DECLARACIÓN USO DE IA
Pedido a claude que me ayudase a hacer el spark_session ya que al realizarlo
sin ayuda obtenía varios errores a la hora de ejecutar los codigos.
En la P4 se quita la parte del conector de MySQL porque esta práctica no lo usa.
'''
def crear_spark_session(app_name="IBEX35_MLOps"):
    builder = (SparkSession.builder
               .appName(app_name)
               .master("local[*]")  # Trabajar en mi ordenador usando todos los nucleos
               # --- Delta Lake (para el feature store del Ej4) ---
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               # --- Igual que en la P0: evita problemas con nombres de equipo con '_' ---
               .config("spark.driver.host", "localhost")
               .config("spark.driver.bindAddress", "127.0.0.1")
               # Datos pequeños en local: 8 particiones de shuffle en vez de 200
               .config("spark.sql.shuffle.partitions", "8"))

    # Añade los .jar de Delta (la 1ª vez los descarga de Maven: necesita internet)
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
