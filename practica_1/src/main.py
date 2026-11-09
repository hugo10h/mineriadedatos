'''
DECLARACIÓN DE USO DE IA

Me he ayduado de la herramienta de IA Claude Sonnet 5.5 para la realización de las prácticas.
He aportado mi idea principal con el funcionamiento de la misma y la IA me ha ayudado a mejorarlas e implementarlas,
y en el peor de los casos a corregirlas y explicarlas.

'''

import os
import shutil
from datetime import date

#dado por gemini para comprobar si funciona
# Ruta del archivo main.py (.../practica_1/src/main.py)
src_dir = os.path.dirname(os.path.abspath(__file__))
# Ruta raíz del módulo practica_1 (.../practica_1)
practica_dir = os.path.dirname(src_dir)
# Ruta del directorio superior MD (.../MD)
root_md_dir = os.path.dirname(practica_dir)

# Configurar HADOOP_HOME apuntando a practica_1
os.environ["HADOOP_HOME"] = practica_dir
os.environ["hadoop.home.dir"] = practica_dir

# Crear carpeta bin dentro de practica_1 si no existe
bin_dir = os.path.join(practica_dir, "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

if not os.path.exists(bin_dir):
    os.makedirs(bin_dir)

# Buscar winutils.exe tanto en MD/ como en practica_1/ para copiarlo automáticamente a bin/
winutils_root = os.path.join(root_md_dir, "winutils.exe")
winutils_prac = os.path.join(practica_dir, "winutils.exe")
winutils_dst = os.path.join(bin_dir, "winutils.exe")

if not os.path.exists(winutils_dst):
    if os.path.exists(winutils_root):
        shutil.copy(winutils_root, winutils_dst)
    elif os.path.exists(winutils_prac):
        shutil.copy(winutils_prac, winutils_dst)


from spark_session import crear_spark_session
from db_connection import obtener_url_conexion, obtener_propiedades_conexion, escribir_tabla, leer_tabla
from ingesta import descargar_historico, descargar_ibex
from pyspark.sql import functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable
from pyspark.sql.window import Window
import yfinance as yf


# Rutas RELATIVAS al proyecto
proyecto_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_LAKE = os.path.join(proyecto_dir, "data", "lake")
CSV_SENTIMIENTO = os.path.join(proyecto_dir, "data", "sentimiento_ibex.csv").replace("\\", "/")
RUTA_BRONZE_IBEX = os.path.join(RUTA_LAKE, "bronze", "ibex").replace("\\", "/")
RUTA_BRONZE_SENT = os.path.join(RUTA_LAKE, "bronze", "sentimiento").replace("\\", "/")
RUTA_BRONZE_EMP = os.path.join(RUTA_LAKE, "bronze", "empresas").replace("\\", "/")

RUTA_SILVER_SENT = os.path.join(RUTA_LAKE, "silver", "sentimiento").replace("\\", "/")
RUTA_SILVER_IBEX = os.path.join(RUTA_LAKE, "silver", "ibex").replace("\\", "/")
RUTA_SILVER_EMP = os.path.join(RUTA_LAKE, "silver", "empresas").replace("\\", "/")

RUTA_GOLD = os.path.join(RUTA_LAKE, "gold").replace("\\", "/")
RUTA_GOLD_FACT = os.path.join(RUTA_GOLD, "fact_cotizaciones").replace("\\", "/")
RUTA_GOLD_DIM_EMP = os.path.join(RUTA_GOLD, "dim_empresas").replace("\\", "/")
RUTA_GOLD_DIM_SENT = os.path.join(RUTA_GOLD, "dim_sentimiento").replace("\\", "/")
RUTA_GOLD_DIM_FECHA = os.path.join(RUTA_GOLD, "dim_fecha").replace("\\", "/")
RUTA_EXPORT_GOLD = os.path.join(proyecto_dir, "data", "export", "gold").replace("\\", "/")

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

df_sent = (spark_session.read.option("header", True).option("sep", ";").option("encoding", "UTF-8").csv(CSV_SENTIMIENTO))
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

######################
###### CLASE 2 #######
######################

print("Ej3")
'''
Convierte las carpetas de bronze (ibex y sentimiento), que ahora son Parquet,
en tablas Delta, sin reescribir los datos. Muestra el historial y comenta qué ha
aparecido en la carpeta (_delta_log).
'''
DeltaTable.convertToDelta(spark_session, f"parquet.`{RUTA_BRONZE_IBEX}`")
DeltaTable.convertToDelta(spark_session, f"parquet.`{RUTA_BRONZE_SENT}`")

dt_ibex = DeltaTable.forPath(spark_session, RUTA_BRONZE_IBEX)
dt_ibex.history().show(truncate = False)
comentar("Al convertir los directorios a Delta, ha aparecido la carpeta _delta_log. Esta carpeta almacena el registro de transacciones (transaction log) en formato JSON, lo cual dota a la tabla de propiedades ACID (Atomicidad, Consistencia, Aislamiento y Durabilidad) y permite el control de versiones sin tener que reescribir los archivos Parquet originales.", "Practica1.pdf (apartado Transaction log y Convertir Parquet a Delta)")


######################################################################################################################

print("Ej4")
'''
Visualiza la versión actual de la base de datos. Ahora vamos a incluir datos
nuevos. Vuelve a descargar los precios del IBEX-35 pero hasta hoy. Haz MERGE
sobre la clave (Ticker, Fecha):
o En la primera ejecución, comprueba que se insertan los días nuevos (el
número de filas sube).
o Ejecútalo una segunda vez y comprueba que pasa.
'''

#guardo la version acutal antes de los cambios para el time travel
v_antes = dt_ibex.history(1).select("version").collect()[0][0]
print("Versión actual de bronze/ibex antes del MERGE:", v_antes)

hoy = date.today().strftime("%Y-%m-%d")
df_ibex_nuevo = descargar_ibex(spark_session, TICKERS_IBEX, FECHA_INICIO, hoy).dropDuplicates(["Ticker", "Fecha"])

def ejecutar_merge():
    #realizo la insercion o actualizacion basada en Ticker
    (dt_ibex.alias("d")
    .merge(df_ibex_nuevo.alias("n"), "d.Ticker = n.Ticker AND d.Fecha = n.Fecha")
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute())

print("Filas ANTES del 1er MERGE:", spark_session.read.format("delta").load(RUTA_BRONZE_IBEX).count())
ejecutar_merge()
print("Filas DESPUÉS del 1er MERGE:", spark_session.read.format("delta").load(RUTA_BRONZE_IBEX).count())

print("Ejecutando MERGE por segunda vez...")
ejecutar_merge()
print("Filas DESPUÉS del 2do MERGE:", spark_session.read.format("delta").load(RUTA_BRONZE_IBEX).count())

comentar("En el primer MERGE el número de filas aumenta porque se insertan los nuevos días hasta la fecha de hoy. En la segunda ejecución las filas se mantienen iguales ya que Delta detecta mediante la clave (Ticker y Fecha) que los datos ya existen, actualizándolos si hubiera cambios. Esto soluciona la duplicación de datos de Parquet logrando una carga idempotente.", "Practica1.pdf (apartado Merge y Ej4)")

################################################################################################################################

print("Ej5-a")
'''
Crea una columna nueva, con los datos de la columna close y sumale 5
(nombre de la columna Close_mas5). Intenta escribir la tabla con esa columna de
más en modo append y comprueba que Delta lo rechaza (error de esquema). 
'''
#obtengo la versión justo antes de corromper el esquema
v_antes_ej5a = dt_ibex.history(1).select("version").collect()[0][0]

df_bronze_actual = spark_session.read.format("delta").load(RUTA_BRONZE_IBEX)
df_con_columna = df_bronze_actual.withColumn("Close_mas5", F.col("Close") + 5)

try:
    df_con_columna.write.format("delta").mode("append").save(RUTA_BRONZE_IBEX)
except Exception as e:
    print(f"Error esperado capturado: {str(e).splitlines()[0]}")

comentar("Delta rechaza la operación por defecto debido al 'schema enforcement' (validación de esquema). La nueva columna 'Close_mas5' no existe en el esquema original de la tabla, impidiendo escrituras accidentales que rompan la consistencia.", "Practica1.pdf (apartado Validación y evolución de esquema)")

#################################################################################################################################

print("Ej5-b")
'''
Ej5b: Repite la escritura del ejercicio anterior con option("mergeSchema", "true") y
comprueba si sigue rechazándolo.
'''

df_con_columna.write.format("delta").mode("append").option("mergeSchema", "true").save(RUTA_BRONZE_IBEX)
print("Columnas tras mergeSchema:", spark_session.read.format("delta").load(RUTA_BRONZE_IBEX).columns)

# Muestro historial y leer con time travel
dt_ibex.history().select("version", "operation", "operationParameters").show(truncate=False)

df_v_antigua = spark_session.read.format("delta").option("versionAsOf", v_antes_ej5a).load(RUTA_BRONZE_IBEX)
print(f"Columnas en versión antigua ({v_antes_ej5a}):", df_v_antigua.columns)
print("Filas de BBVA en versión antigua:", df_v_antigua.filter("Ticker = 'BBVA.MC'").count())

comentar("Al usar versionAsOf podemos leer el estado exacto de la tabla antes de añadir la columna y borrar las filas, comprobando que Delta mantiene accesibles los datos eliminados sin alterar el estado actual de la tabla principal.", "Practica1.pdf (apartado Time travel)")

#################################################################################################################################

print("Ej5-c")
'''
Elimina con DELETE las filas del ticker BBVA. Muestra las filas antes y
después.
'''

#Elimino las dilas de un ticker concreto
print("Filas ANTES del DELETE:", spark_session.read.format("delta").load(RUTA_BRONZE_IBEX).count())
dt_ibex.delete("Ticker = 'BBVA.MC'")
print("Filas DESPUÉS del DELETE:", spark_session.read.format("delta").load(RUTA_BRONZE_IBEX).count())

#####################################################################################################################################

print("Ej5-d")
'''
 Muestra el history() (verás las versiones de los Ej5b y Ej5c) y lee con
versionAsOf a la versión de antes del Ej5a. Comprueba si todo esta igual.
'''

#muestro el historial y leeo con time travel
dt_ibex.history().select("version", "operation", "operationParameters").show(truncate=False)

df_v_antigua = spark_session.read.format("delta").option("versionAsOf", v_antes_ej5a).load(RUTA_BRONZE_IBEX)
print(f"Columnas en versión antigua ({v_antes_ej5a}):", df_v_antigua.columns)
print("Filas de BBVA en versión antigua:", df_v_antigua.filter("Ticker = 'BBVA.MC'").count())

##############################################################################################################################################3

print("Ej6")
'''
VersionAsOf lee la versión antigua (demuestra que el dato sigue recuperable),
pero no cambia la tabla actual. Deja la tabla realmente como estaba en el EJ4,
usando RESTORE
'''

# Restauro físicamente la tabla a la versión anterior
dt_ibex.restoreToVersion(v_antes_ej5a)

print(f"Tabla restaurada a la versión {v_antes_ej5a}")
df_restaurado = spark_session.read.format("delta").load(RUTA_BRONZE_IBEX)
print("Columnas actuales tras RESTORE:", df_restaurado.columns)
print("Filas actuales de BBVA tras RESTORE:", df_restaurado.filter("Ticker = 'BBVA.MC'").count())
comentar("RESTORE devuelve la tabla activa al estado de la versión indicada deshaciendo el DELETE y el append con evolución de esquema. A diferencia de versionAsOf que solo lee, RESTORE crea un nuevo commit dejando el dato revertido pero manteniendo todo el historial intacto en el log.", "Practica1.pdf (apartado Restore)")

########################################################################################################################################

######################
###### CLASE 3 #######
######################

############################################################################################################################
print("Ej7")
'''
Analiza los datos e incluye una reflexión de como podrías tener en cuenta la
información del experto en los datos que puedes obtener de yahoo finance.
(Investiga que datos tiene la API de yahoo u otras interesantes). Justifícalo en el
pdf
'''

ESQUEMA_EMPRESAS = StructType([
    StructField("Ticker", StringType(), False),
    StructField("Sector", StringType(), True),
    StructField("Industria", StringType(), True),
    StructField("Nombre", StringType(), True),
    StructField("Capitalizacion", DoubleType(), True),
    StructField("Beta", DoubleType(), True),
])

def obtener_sectores_yahoo(tickers):
    datos = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap = info.get("marketCap")
            beta = info.get("beta")
            datos.append((t,
                          info.get("sector") or "Desconocido",
                          info.get("industry") or "Desconocido",
                          info.get("shortName") or t,
                          float(cap) if cap is not None else None,
                          float(beta) if beta is not None else None))
        except Exception:
            datos.append((t, "Desconocido", "Desconocido", t, None, None))
    return datos

comentar("La API de Yahoo Finance contiene el objeto 'info' para cada ticker, el cual incluye el atributo 'sector'. Este dato es fundamental como dimensión descriptiva, ya que actúa como clave foránea para relacionar las cotizaciones individuales de las empresas con el índice de sentimiento sectorial proporcionado en el CSV.", "Practica1_2.pdf (apartado Silver: limpiar, tipar y validar y Ej7)")

###########################################################################################################################################

print("Ej8")
'''
 Construye la capa silver. Limpia la base de datos sentimiento y guárdalo en
data/lake/silver/sentimiento. Los sectores del CSV están en español. Justifica la
construcción/modificación de los datos en el pdf. Muestra cuántas filas tiene
cada tabla.
'''

def procesar_silver_sentimiento(spark, ruta_origen, ruta_destino):
    df_origen = spark.read.format("delta").load(ruta_origen)

    df_final = (df_origen
                .withColumn("Fecha", F.to_date("Fecha", "yyyy-MM-dd"))
                .withColumn("Sector", F.initcap(F.trim(F.col("Sector"))))
                .withColumn("Sentimiento", F.regexp_replace("Sentimiento", ",", ".").cast("double"))
                .dropna(subset=["Fecha", "Sector", "Sentimiento"])
                .dropDuplicates(["Fecha", "Sector"]))

    # Comprobación: sectores que quedan tras normalizar
    df_final.select("Sector").distinct().show(50, truncate=False)

    df_final.write.format("delta").mode("overwrite").save(ruta_destino)
    return df_final

df_silver_sent = procesar_silver_sentimiento(spark_session, RUTA_BRONZE_SENT, RUTA_SILVER_SENT)
print(f"Filas en Silver Sentimiento: {df_silver_sent.count()}")

comentar("El CSV ya venía en formato largo, así que no hizo falta unpivot. Se convirtió Fecha a date, "
         "Sentimiento a double (cambiando coma por punto por si acaso), se normalizó Sector con trim e "
         "initcap para que coincida con los nombres de Yahoo Finance, y se eliminaron nulos y duplicados "
         "por (Fecha, Sector).", "Practica1_2.pdf (apartado Silver: limpiar, tipar y validar y Ej8)")
##########################################################################################################

print("Ej 9")
'''
 Deja los datos de la api de yahoo finance limpos (si es necesario) y guárdalos
en data/lake/silver/ibex. Si necesitas más datos (Ej7) guarda en una dimensión (o
varias) aparte tanto en bronce (datos en crudo) como en silver (datos corregidos)
la información. Muestra la/s fila/s 5.
'''

def procesar_silver_ibex(spark, ruta_origen, ruta_destino):
    #limpieza eliminando duplicados y nulos
    df = (
        spark.read.format("delta").load(ruta_origen).dropna(subset=["Close", "Ticker", "Fecha"]).dropDuplicates(["Ticker", "Fecha"])
    )

    df.write.format("delta").mode("overwrite").save(ruta_destino)
    return df

def procesar_dimension_empresas(spark, tickers, ruta_bronze, ruta_silver):
    #Limpieza de la dimension de las empresas
    datos_empresas = obtener_sectores_yahoo(tickers)
    df_emp = spark.createDataFrame(datos_empresas, ESQUEMA_EMPRESAS)

    #presistencia en bronze
    df_emp.write.mode("overwrite").parquet(ruta_bronze)
    DeltaTable.convertToDelta(spark, f"parquet.`{ruta_bronze}`")

    #procesamiento y guardado en silver
    df_emp_silver = spark.read.format("delta").load(ruta_bronze).dropDuplicates(["Ticker"])
    df_emp_silver.write.format("delta").mode("overwrite").save(ruta_silver)
    return df_emp_silver

df_silver_ibex = procesar_silver_ibex(spark_session, RUTA_BRONZE_IBEX, RUTA_SILVER_IBEX)
df_silver_ibex.show(5)

df_silver_empresa = procesar_dimension_empresas(spark_session, TICKERS_IBEX, RUTA_BRONZE_EMP, RUTA_SILVER_EMP)
df_silver_empresa.show(5)

print("Filas bronze ibex:", spark_session.read.format("delta").load(RUTA_BRONZE_IBEX).count())
print("Filas bronze sentimiento:", spark_session.read.format("delta").load(RUTA_BRONZE_SENT).count())
print("Filas silver ibex:", df_silver_ibex.count())
print("Filas silver sentimiento:", df_silver_sent.count())
print("Filas silver empresas:", df_silver_empresa.count())

###################################################################################################################3

print("Ej10")
'''
Integra todas las fuentes en la capa y guárdala en data/lake/gold
'''
def crear_dim_fecha(df_ibex):
    # dimensión Fecha: un registro por día de cotización, con atributos para agrupar en Power BI
    return (df_ibex.select("Fecha").distinct()
            .withColumn("Anio", F.year("Fecha"))
            .withColumn("Trimestre", F.quarter("Fecha"))
            .withColumn("Mes", F.month("Fecha"))
            .withColumn("Semana", F.weekofyear("Fecha"))
            .withColumn("Dia_Semana", F.dayofweek("Fecha")))


def integrar_capa_gold(spark, ruta_ibex, ruta_sentiminento, ruta_empresa):
    #une las dimensiones procesadas en la capa silver para generar la tabla
    df_ibex = spark.read.format("delta").load(ruta_ibex)
    df_sent = spark.read.format("delta").load(ruta_sentiminento)
    df_emp = spark.read.format("delta").load(ruta_empresa)

    #construimos tabla fact con LEFT Joins
    df_fact = (
        df_ibex.join(df_emp.select("Ticker", "Sector"), on="Ticker", how="left")
    )

    #guardamos la tabla de hechos y las tablas de dimensiones
    df_fact.write.format("delta").mode("overwrite").save(RUTA_GOLD_FACT)
    df_emp.write.format("delta").mode("overwrite").save(RUTA_GOLD_DIM_EMP)
    df_sent.write.format("delta").mode("overwrite").save(RUTA_GOLD_DIM_SENT)
    crear_dim_fecha(df_ibex).write.format("delta").mode("overwrite").save(RUTA_GOLD_DIM_FECHA)
    return df_fact

df_gold_fact = integrar_capa_gold(spark_session, RUTA_SILVER_IBEX, RUTA_SILVER_SENT, RUTA_SILVER_EMP)
print(f"Hechos en capa Gold procesados: {df_gold_fact.count()} filas")

####################################################################################################3

print("Ej11")
'''
 Con un anti-join, muestra (a) los precios que se quedan sin
sentimiento y (b) el sentimiento sin precios. Comenta que sucede y como lo
solucionas.
'''

def analisis_calidad_antijoin(spark, ruta_fact, ruta_sent):
    """Analiza registros no cruzados entre hechos y dimensiones."""
    df_fact = spark.read.format("delta").load(ruta_fact)
    df_sent = spark.read.format("delta").load(ruta_sent)
    
    precios_sin_sentimiento = df_fact.join(df_sent, on=["Sector", "Fecha"], how="left_anti")
    sentimiento_sin_precios = df_sent.join(df_fact, on=["Sector", "Fecha"], how="left_anti")
    
    print("Rango de fechas del sentimiento:")
    df_sent.agg(F.min("Fecha"), F.max("Fecha")).show()
    print("Rango de fechas de las cotizaciones:")
    df_fact.agg(F.min("Fecha"), F.max("Fecha")).show()

    print(f"Cotizaciones sin métrica de sentimiento: {precios_sin_sentimiento.count()}")
    precios_sin_sentimiento.select("Ticker", "Sector", "Fecha").show(5)
    print(f"Sentimiento sin empresas cotizando: {sentimiento_sin_precios.count()}")
    sentimiento_sin_precios.show(5)

analisis_calidad_antijoin(spark_session, RUTA_GOLD_FACT, RUTA_GOLD_DIM_SENT)

comentar(
    "Los precios sin sentimiento son las cotizaciones posteriores a la última fecha del CSV del experto "
    "(el MERGE del Ej4 trajo precios hasta hoy). El sentimiento sin precios se debe a que el CSV empieza en "
    "2021 y los precios en octubre de 2024, y a que el sector Crypto no tiene ninguna empresa del IBEX. "
    "No se rellenan los nulos: arrastrar el último valor inventaría un dato que el experto no emitió, "
    "así que en Power BI esos días aparecen vacíos. Las filas sin cruzar se conservan gracias al LEFT JOIN.",
    "Practica1_2.pdf (apartado Anti-join left_anti)"
)

############################################################################################################

print("Ej12")
'''
Ej12 (opcional): Vuelve al estudio realizado en el Ej7 y piensa que datos te gustaría
ver en el dashboard (adicionales a los datos tal cual) que se hará en power BI (por
ejemplo, la variación de precio respecto al día anterior). Realiza al menos dos
operaciones/cálculos y guárdalos en una nueva columna
'''

def enriquecer_metricas(df):
    """Calcula variaciones intradía y el porcentaje de retorno diario usando ventanas de Spark."""
    ventana_ticker = Window.partitionBy("Ticker").orderBy("Fecha")
    return (
        df
        .withColumn("Variacion_Intradia", F.round(F.col("Close") - F.col("Open"), 4))
        .withColumn(
            "Retorno_Diario_Pct",
            F.round(
                ((F.col("Close") - F.lag("Close").over(ventana_ticker)) / F.lag("Close").over(ventana_ticker)) * 100, 
                2
            )
        )
    )

df_gold_enriquecido = enriquecer_metricas(df_gold_fact)
df_gold_enriquecido.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(RUTA_GOLD_FACT)
df_gold_enriquecido.select("Ticker", "Fecha", "Close", "Variacion_Intradia", "Retorno_Diario_Pct").show(5)

#################################################################################################

print("Ej13")
'''
 Exporta la gold a Parquet en data/export/gold, lista para Power BI. (Tienes que
elegir como lo vas a guardar). Investiga más sobre la forma de estrella para
guardar los datos (Cuanto más eficiente lo hagas, mejor funcionará la siguiente
práctica. Pregunta a tu profesor si tienes dudas una vez tengas el modelo).
Justifícalo en UN párrafo en el pdf.
'''
def exportar_para_powerbi(spark, ruta_origen, ruta_exportacion):
    #esportar las tablas GOLD en formato parquet PowerBi
    df_hechos = spark.read.format("delta").load(os.path.join(ruta_origen, "fact_cotizaciones"))
    df_hechos.write.mode("overwrite").parquet(os.path.join(ruta_exportacion, "fact_cotizaciones"))

    df_emp = spark.read.format("delta").load(os.path.join(ruta_origen, "dim_empresas"))
    df_emp.write.mode("overwrite").parquet(os.path.join(ruta_exportacion, "dim_empresas"))

    df_sent = spark.read.format("delta").load(os.path.join(ruta_origen, "dim_sentimiento"))
    df_sent.write.mode("overwrite").parquet(os.path.join(ruta_exportacion, "dim_sentimiento"))

    df_fecha = spark.read.format("delta").load(os.path.join(ruta_origen, "dim_fecha"))
    df_fecha.write.mode("overwrite").parquet(os.path.join(ruta_exportacion, "dim_fecha"))


exportar_para_powerbi(spark_session, RUTA_GOLD, RUTA_EXPORT_GOLD)

comentar(
    "Se exporta el modelo en estrella: una tabla de hechos central (cotización diaria por empresa, con el "
    "sentimiento de su sector y las métricas calculadas) unida por Ticker y Fecha a las dimensiones empresa "
    "(sector, industria, capitalización, beta) y fecha (año, trimestre, mes, semana). Las dimensiones "
    "evitan repetir texto en cada fila del hecho y Power BI las usa directamente como filtros y ejes. "
    "Se conserva además dim_sentimiento como consulta del índice completo del experto, incluida la parte "
    "sin cotizaciones. Se exporta en Parquet por su lectura columnar.",
    "Practica1_2.pdf (apartado El modelo en estrella y Ej13)"
)