from spark_session import crear_spark_session
from db_connection import obtener_url_conexion, obtener_propiedades_conexion, escribir_tabla
from pyspark.sql.types import StructType, StructField, DateType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, avg
import os
from pyspark.sql.window import Window


spark_session = crear_spark_session()

# ---------------------------------------------------------------------
# CONFIGURACION DE LA BASE DE DATOS
# Unico sitio que hay que tocar para ejecutar esto en otro servidor.
# ---------------------------------------------------------------------

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "1433"))
DB_INSTANCE = os.environ.get("DB_INSTANCE") or None
DB_NAME = os.environ.get("DB_NAME", "IBEX35")
DB_USER = os.environ.get("DB_USER", "spark_user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Ibex35Pass1!")

url_bd = obtener_url_conexion(DB_HOST, DB_PORT, DB_NAME, DB_INSTANCE)
propiedades_bd = obtener_propiedades_conexion(DB_USER, DB_PASSWORD)


proyecto_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = os.path.join(
    proyecto_dir,
    "data",
    "ibex35_close-2024.csv"
)

if not os.path.isfile(path):
    raise FileNotFoundError(
        "No se encuentra el CSV. Debe existir en: " + path
    )

df = (spark_session.read
      .option("header", True)
      .option("sep", ";")
      .option("dateFormat", "dd/MM/yyyy")
      .csv(path))


print("Datos sin tratar:")
escribir_tabla(df, "Datos2024", url_bd, propiedades_bd)


################################
###########EJERCICIOS###########
################################


#1A
print("Ej 1-a")
from pyspark.sql.functions import to_date, col
df.printSchema()
df = df.withColumn(
    "Fecha",
    to_date("Fecha", "dd/MM/yyyy")
    
)
for columna in df.columns:
    if columna != "Fecha":
        df = df.withColumn(columna, col(f"`{columna}`").cast("double"))
df.printSchema()
df.show(6)


#1B
print("Ej 1-b")
for columna in df.columns:
    nuevo_nombre = columna.replace(".MC", "")
    df = df.withColumnRenamed(columna, nuevo_nombre)
df.show(6)


#1C
print("Ej 1-c")

'''
DECLARACION SO DE IA

Utilizada la inteligencia artificial para obtener todos los nombres de las empresas
'''
schema_1c = StructType([
    StructField("Fecha", DateType(), True),
    StructField("Iberdrola", DoubleType(), True),
    StructField("Repsol", DoubleType(), True),
    StructField("Naturgy", DoubleType(), True),
    StructField("Endesa", DoubleType(), True),
    StructField("Enagas", DoubleType(), True),
    StructField("Redeia", DoubleType(), True),
    StructField("Santander", DoubleType(), True),
    StructField("BBVA", DoubleType(), True),
    StructField("CaixaBank", DoubleType(), True),
    StructField("Bankinter", DoubleType(), True),
    StructField("Sabadell", DoubleType(), True),
    StructField("Unicaja", DoubleType(), True),
    StructField("Mapfre", DoubleType(), True),
    StructField("ACS", DoubleType(), True),
    StructField("Acciona", DoubleType(), True),
    StructField("AccionaEnergia", DoubleType(), True),
    StructField("Acerinox", DoubleType(), True),
    StructField("ArcelorMittal", DoubleType(), True),
    StructField("Sacyr", DoubleType(), True),
    StructField("Cellnex", DoubleType(), True),
    StructField("Telefonica", DoubleType(), True),
    StructField("AENA", DoubleType(), True),
    StructField("Ferrovial", DoubleType(), True),
    StructField("Inditex", DoubleType(), True),
    StructField("Amadeus", DoubleType(), True),
    StructField("IAG", DoubleType(), True),
    StructField("Grifols", DoubleType(), True),
    StructField("Fluidra", DoubleType(), True),
    StructField("Solaria", DoubleType(), True),
    StructField("Rovi", DoubleType(), True),
    StructField("Logista", DoubleType(), True),
    StructField("Indra", DoubleType(), True),
    StructField("Melia", DoubleType(), True),
    StructField("Puig", DoubleType(), True),
    StructField("Colonial", DoubleType(), True),
    StructField("Merlin", DoubleType(), True)
])

df_1c = (spark_session.read
         .option("header", False)
         .option("sep", ";")
         .option("dateFormat", "dd/MM/yyyy")
         .schema(schema_1c)
         .csv(path))

df_1c = df_1c.filter(col("Fecha").isNotNull())

df_1c.printSchema()
df_1c.show(6)



#Ej 2a
print("Ej2-a")

filas_antes = df.count()
df = df.dropDuplicates().orderBy("Fecha")
filas_despues = df.count()
print("Número de filas eliminadas: ", filas_antes - filas_despues)

empresas = [c for c in df.columns if c != "Fecha"]
no_nulso = df.select([count(col(c)).alias(c) for c in empresas]).first().asDict()
columnas_sin_info = [c for c, n in no_nulso.items() if n == 0]
df = df.drop(*columnas_sin_info)

print("Empresas con informacion: ", len(df.columns) - 1)

#Ej 2b
print("Ej-2b")

from pyspark.sql.functions import min as spark_min
from pyspark.sql.functions import max as spark_max


fecha_incial = df.select(spark_min("Fecha")).first()[0]
fecha_final = df.select(spark_max("Fecha")).first()[0]

dias = df.select(count("Fecha")).first()[0]

print("Fecha inicial: ", fecha_incial)
print("Fecha final: ", fecha_final)
print("Dias : ", dias)

print("Comentario: El periodo de datos corresponde al año 2024, por lo que el resultado es coherente con lo esperado.")
print("No considero necesario buscar datos adicionales si el objetivo es analizar únicamente el periodo disponible en 2024.")


#Ej3
print("Ej-3")

df = df.withColumnRenamed("Fecha", "Dia")
df.show(10, truncate=False)

empresas = [c for c in df.columns if c != "Dia"]

resultados = []

for empresa in empresas:
    media = df.select(avg(col(empresa))).first()[0]
    min = df.select(spark_min(col(empresa))).first()[0]
    max = df.select(spark_max(col(empresa))).first()[0]

resultados.append((empresa, media, max, min))

print("Empresa - Media anual - Max anual - Min anual")

for resultado in resultados:
    print(resultado)

# Deficiency Notice UNI

df = df.withColumn(
    "Deficiency Notice UNI",
    when(col("UNI") < 1, True).otherwise(False)
)

df.show(100, truncate=False)

#Ej 4
print("Ej-4")


filas_variacion = []
for empresa in empresas:
    fila_inicial = df.filter(F.col(empresa).isNotNull()).orderBy("Dia").select(empresa).first()
    fila_final = df.filter(F.col(empresa).isNotNull()).orderBy(F.col("Dia").desc()).select(empresa).first()
    if fila_inicial is None or fila_final is None:
        continue
    valor_inicial = fila_inicial[0]
    valor_final = fila_final[0]
    variacion = ((valor_final - valor_inicial) / valor_inicial) * 100
    filas_variacion.append((empresa, round(valor_inicial, 2), round(valor_final, 2), round(variacion, 2)))

df_variacion = spark_session.createDataFrame(
    filas_variacion,
    ["Empresa", "Valor inicial", "Valor final", "Variacion Anual"]
)

df_variacion = df_variacion.withColumn(
    "Clasificacion",
    F.when(F.col("Variacion Anual") >= 15, "Subida Fuerte")
     .when(F.col("Variacion Anual") > 1, "Subida")
     .when(F.col("Variacion Anual") >= -1, "Neutra")
     .when(F.col("Variacion Anual") > -15, "Bajada")
     .otherwise("Bajada Fuerte")
)

df_variacion.show(len(empresas), truncate=False)

# Ej5
print("Ej5")

for empresa in empresas:
    q1, q2, q3 = df.approxQuantile(empresa, [0.25, 0.5, 0.75], 0.01)
    df = df.withColumn(
        f"{empresa}Cuartil",
        F.when(F.col(empresa).isNull(), None)
         .when(F.col(empresa) <= q1, "q1")
         .when(F.col(empresa) <= q2, "q2")
         .when(F.col(empresa) <= q3, "q3")
         .otherwise("q4")
    )

df.show(1)
df.select("Dia", "AENA", "AENACuartil", "BBVA", "BBVACuartil").show(df.count(), truncate=False)

#Ej 6
print("Ej-6")

ventana_dia = Window.orderBy("Dia")

for empresa in empresas:
    col_lag = F.lag(F.col(empresa)).over(ventana_dia)
    variacion_diaria = ((F.col(empresa) - col_lag) / col_lag) * 100
    df = df.withColumn(
        f"{empresa}CambioSignificativo",
        F.when(F.abs(variacion_diaria) > 8, F.round(variacion_diaria, 2)).otherwise(F.lit("-"))
    )

df_indexado = df.withColumn("num_fila", F.row_number().over(ventana_dia))
df_indexado.filter(F.col("num_fila") == 15).drop("num_fila").show(1, truncate=False)


'''
Declaracion USO  de IA

Usado para pedir el texto y la referencia de los prints
'''
print(
    "Investigacion: la mayoria de los cambios superiores al 8% en un solo dia "
    "corresponden a Grifols, que sufrio un desplome del 25,91% el 9/01/2024 "
    "tras un informe del fondo bajista Gotham City Research que acusaba a la "
    "compañia de manipular su ratio de deuda; el valor siguio siendo muy volatil "
    "el resto del año por la investigacion de la CNMV y los litigios con Gotham."
)

print("Referencia: https://valenciaplaza.com/valenciaplaza/grifols-un-ano-despues-cotizacion-sigue-34-por-debajo-tras-gotham-opa-brookfield")