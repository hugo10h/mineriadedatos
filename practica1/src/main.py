import os
import shutil
from datetime import date

import os
import shutil
from datetime import date

from spark_session import crear_spark_session
from db_connection import obtener_url_conexion, obtener_propiedades_conexion, escribir_tabla, leer_tabla
from ingesta import descargar_historico, descargar_ibex
from pyspark.sql import functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable

# Rutas RELATIVAS al proyecto
proyecto_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_LAKE = os.path.join(proyecto_dir, "data", "lake")
RUTA_BRONZE_IBEX = os.path.join(RUTA_LAKE, "bronze", "ibex").replace("\\", "/")
RUTA_BRONZE_SENT = os.path.join(RUTA_LAKE, "bronze", "sentimiento").replace("\\", "/")
RUTA_CSV_SENT = os.path.join(proyecto_dir, "data", "sentimiento_ibex.csv")

# Para poder repetir el programa desde cero: borra el lake de la ejecución anterior
if os.path.exists(RUTA_LAKE):
    shutil.rmtree(RUTA_LAKE)

spark_session = crear_spark_session()

url_bd = obtener_url_conexion("localhost", 3306, "ibex35")
propiedades_bd = obtener_propiedades_conexion("spark_user", "Ibex35Pass1!")

FECHA_INICIO = "2024-10-01"
FECHA_FIN = "2026-10-01"
TICKERS_IBEX = [
    "IBE.MC", "REP.MC", "NTGY.MC", "ELE.MC", "ENG.MC", "RED.MC",
    "SAN.MC", "BBVA.MC", "CABK.MC", "BKT.MC", "SAB.MC", "UNI.MC", "MAP.MC",
    "ACS.MC", "ANA.MC", "ANE.MC", "ACX.MC", "MTS.MC", "SCYR.MC",
    "CLNX.MC", "TEF.MC",
    "AENA.MC", "FER.MC", "ITX.MC", "AMS.MC", "IAG.MC", "MEL.MC",
    "GRF.MC", "FDR.MC", "SLR.MC", "ROVI.MC", "LOG.MC", "IDR.MC", "PUIG.MC",
    "COL.MC", "MRL.MC",
]


def comentar(texto, referencia):
    # Formato de comentario que pide la normativa
    print("Comentario/Reflexion/Investigacion")
    print(texto)
    print("Referencia/s")
    print(referencia)


######################
###### CLASE 1 #######
######################

print("Ej1-a")
''' Crea una función que permita descargar el histórico de una acción concreta,
facilitando una fecha de inicio y una de fin, y lo almacene en un dataframe de
pyspark. Define para cada columna si permite nulos y por qué. '''

df_bbva = descargar_historico(spark_session, "BBVA.MC", FECHA_INICIO, FECHA_FIN)
df_bbva.printSchema()
#DECLARACIÓN USO DE IA --> Pedido a Claude (comentar)
comentar("Fecha y Ticker NO admiten nulos: son la clave de cada fila (Ticker + Fecha) y sin ellas "
         "no se puede ordenar ni cruzar. Open, High, Low, Close y Volume SI admiten nulos: en bronze "
         "se guarda el dato crudo y Yahoo puede dejar huecos; limpiarlos es trabajo de silver. "
         "Dividends y Stock_Splits NO admiten nulos: Yahoo pone 0.0 cuando no hay evento.",
         "Practica1.pdf (apartado Practica: campos de yfinance)")

##############################################################################################################

print("Ej1-b")
''' Usando la función anterior, descarga los últimos 2 años (último día 1 de
octubre de 2026) de todas las empresas del IBEX-35 y guárdalo crudo en la capa
bronce (lo veremos más adelante que significa), en formato Parquet:
 df.write.mode("overwrite").parquet("data/lake/bronze/ibex") '''

df_ibex = descargar_ibex(spark_session, TICKERS_IBEX, FECHA_INICIO, FECHA_FIN)
df_ibex.write.mode("overwrite").parquet(RUTA_BRONZE_IBEX)

##############################################################################################################

print("Ej1-c")
''' Lee de vuelta el Parquet que acabas de escribir y muestra el
esquema y las 5 primeras filas. A continuación, lee solo las columnas Ticker, Date
y Close (column pruning) y muéstralas. Comenta qué ventaja tiene leer solo esas
columnas frente a leerlas todas'''

df_lectura = spark_session.read.parquet(RUTA_BRONZE_IBEX)
df_lectura.printSchema()
df_lectura.show(5)
spark_session.read.parquet(RUTA_BRONZE_IBEX).select("Ticker", "Fecha", "Close").show(5)
#DECLARACIÓN USO DE IA --> Pedido a Claude (comentar)
comentar("Parquet guarda cada columna por separado (column chunk), asi que al pedir solo "
         "Ticker, Fecha y Close Spark lee unicamente esas columnas del disco y se salta el resto. "
         "Hay menos lectura de disco, menos memoria y la consulta es mas rapida que leyendo todas.",
         "Practica1.pdf (apartado La estructura de un fichero Parquet: column pruning)")

###############################################################################################################


print("Ej2-a")
'''Guarda el fichero sentimiento_ibex.csv en un DataFrame de PySpark. No abrir
el csv. Observa y comenta que todas las columnas llegan como texto (un CSV no
guarda los tipos)'''

df_sent = (spark_session.read.option("header", True).option("sep", ";").option("encoding", "UTF-8").csv(RUTA_CSV_SENT))
df_sent.printSchema()
df_sent.show(5)
#DECLARACIÓN USO DE IA --> Pedido a Claude (comentar)
comentar("Todas las columnas llegan como string: un CSV es texto plano y no guarda tipos. "
         "Como no usamos inferSchema, Spark no intenta adivinarlos. Los tipos se corregiran "
         "en silver (fechas a date, numeros a double).",
         "Practica1.pdf (Ej2a)")

################################################################################################################


print("Ej-2b")
'''
Guarda el sentimiento crudo en la capa bronze en formato Parquet, en su
propia carpeta (data/lake/bronze/sentimiento). Muestra cuántas filas se han
escrito.
'''

df_sent.write.mode("overwrite").parquet(RUTA_BRONZE_SENT)
print("Filas escritas en bronze/sentimiento", df_sent.count())

##################################################################################################################

print("Ej-2c")
'''
Vuelve a ejecutar la carga del Ej2b pero esta vez en modo append.
Cuenta las filas de la carpeta antes y después y comprueba que pasa. Comenta que
ocurre. 
'''

filas_antes = spark_session.read.parquet(RUTA_BRONZE_SENT).count()
df_sent.write.mode("append").parquet(RUTA_BRONZE_SENT)
filas_despues = spark_session.read.parquet(RUTA_BRONZE_SENT).count()
print("Antes: ", filas_antes, "| Despues: ", filas_despues)
comentar("Con append Parquet solo anade ficheros nuevos a la carpeta, no comprueba si los datos "
         "ya existian. Las filas se duplican (el conteo se dobla) porque Parquet no tiene claves "
         "ni transacciones, y la carga no es idempotente. Delta con MERGE soluciona esto.",
         "Practica1.pdf (apartado Delta Lake: Merge)")

####################################################################################################################

#DEJAR LIMPIO BRONZE_SENTIMIENTO PARA EJERCICIOS POSTERIORES
df_sent.write.mode("overwrite").parquet(RUTA_BRONZE_SENT)

####################################################################################################################

