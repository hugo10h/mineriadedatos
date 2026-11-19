'''
DECLARACIÓN DE USO DE IA

Me he ayudado de la herramienta de IA Claude para la realización de las prácticas.
He aportado mi idea principal con el funcionamiento de la misma y la IA me ha ayudado a mejorarlas e implementarlas,
y en el peor de los casos a corregirlas y explicarlas.

PRÁCTICA 4: Aprendizaje no supervisado y MLOps con Spark  (Clase 1: el sistema MLOps)

main.py es el CONTROLADOR (patrón MVC):
    - Modelo : datos.py, modelo_kmeans.py, metricas.py, entrenamiento.py, busqueda_hyperopt.py,
               seguimiento_mlflow.py, feature_store.py, exportacion.py
    - Vista  : vista.py
    - Control: este fichero. Cada ejercicio solo LLAMA a funciones del modelo y MUESTRA el resultado.
'''
import random

import numpy as np

import config as cfg
from spark_session import crear_spark_session
from datos import cargar_gold, datos_empresa
from entrenamiento import entrenar_evaluar_registrar
from busqueda_hyperopt import buscar_mejor_k
from seguimiento_mlflow import (configurar_mlflow, listar_experimentos, mejor_run,
                                obtener_runs_ordenadas)
from feature_store import guardar_features, leer_features, versiones_disponibles
from modelo_kmeans import centros
from exportacion import exportar_clusters
import vista

# ---------------------------------------------------------------------------
# Semillas globales: random de Python y numpy (pandas, Hyperopt internamente...)
# Spark NO usa estas: cada algoritmo de MLlib tiene su propio parámetro seed.
# ---------------------------------------------------------------------------
random.seed(cfg.SEED_GLOBAL)
np.random.seed(cfg.SEED_GLOBAL)


######################
###### CLASE 1 #######
######################

def ej1(df_bbva):
    vista.cabecera("Ej1", "Pipeline VectorAssembler + KMeans con MLflow (BBVA, Open y Close)")
    '''
    Crea una Pipeline con MLflow en Spark que utilice VectorAssembler y KMeans, con el objetivo de
    que pueda clasificar los datos que se proporcionen. Utiliza únicamente los datos de BBVA
    correspondientes al Valor de Apertura y Valor de Cierre, con el fin de clasificar cada día según
    su comportamiento en el mercado. Muestra el número de agrupaciones correcto. (No usar Hyperopt).
    '''
    configurar_mlflow(cfg.MLRUNS_DIR, cfg.EXPERIMENTO_EJ1)

    df_bbva.printSchema()
    df_bbva.show(5)
    print("Filas BBVA:", df_bbva.count())

    # Sin Hyperopt: se prueban "a mano" varios k y cada uno queda registrado como una run
    resultados = [entrenar_evaluar_registrar(df_bbva, cfg.TICKER_EJ1, cfg.COLUMNAS_EJ1,
                                             k, cfg.SEED_MODELO)
                  for k in cfg.K_CANDIDATOS_EJ1]
    vista.tabla_k(resultados)

    # Número de agrupaciones correcto = el k con MAYOR silhouette
    mejor = max(resultados, key=lambda r: r["silhouette"])
    print(f"\nNúmero de agrupaciones correcto (máxima silhouette): k = {mejor['k']} "
          f"(silhouette = {mejor['silhouette']:.4f})")
    mejor["predicciones"].select("Fecha", "Open", "Close", "prediction").show(5)
    vista.resumen_clusters(mejor["predicciones"], centros(mejor["modelo"]))

    vista.comentar(
        "Se construye un Pipeline VectorAssembler -> KMeans (Open y Close están en euros y en la "
        "misma escala, por eso no hace falta StandardScaler). Se entrena para k=2..8, cada intento "
        "se registra en MLflow (parámetros y métricas) y se elige el k con mayor silhouette, que "
        "mide a la vez lo juntos que están los puntos de su grupo y lo separados que están del "
        "grupo vecino. La SSE también se guarda, pero siempre baja al subir k (método del codo).",
        "Enunciado-Practica4.pdf (KMeans, Pipeline en Spark, MLflow); teoriaCOMPLETO.pdf "
        "(Selección del número de agrupaciones)")
    return mejor


def ej2(df_bbva):
    vista.cabecera("Ej2", "Búsqueda del mejor k con Hyperopt + registro en MLflow")
    '''
    Partiendo del Ej1, implementa una búsqueda automatizada de hiperparámetros con Hyperopt para
    encontrar el mejor número de clusters k (al menos 5 valores) de KMeans, utilizando otra métrica
    distinta a la anterior, y registra cada intento en MLflow. Muestra el número de agrupaciones
    seleccionado por Hyperopt. Fija las dos seeds y explica en un comentario qué fija cada una.
    '''
    configurar_mlflow(cfg.MLRUNS_DIR, cfg.EXPERIMENTO_EJ2)

    # SEMILLA 1 -> cfg.SEED_MODELO (KMeans(seed=...)): fija la inicialización aleatoria de los
    #   centroides de KMeans. Mismo k + mismos datos = mismos clusters en cada entrenamiento.
    # SEMILLA 2 -> cfg.SEED_HYPEROPT (rstate de fmin): fija qué valores de k decide probar TPE
    #   y en qué orden. Misma búsqueda = mismos intentos = mismo k elegido.
    mejor_k, trials = buscar_mejor_k(df_bbva, cfg.TICKER_EJ1, cfg.COLUMNAS_EJ1,
                                     cfg.K_MIN, cfg.K_MAX, cfg.MAX_EVALS,
                                     cfg.SEED_MODELO, cfg.SEED_HYPEROPT)
    print(f"\nNúmero de agrupaciones seleccionado por Hyperopt: k = {mejor_k} "
          f"(Davies-Bouldin = {min(trials.losses()):.4f}, {len(trials.trials)} intentos)")

    # Modelo final con el k ganador: esta run sí guarda el PipelineModel como artefacto
    final = entrenar_evaluar_registrar(df_bbva, cfg.TICKER_EJ1, cfg.COLUMNAS_EJ1, mejor_k,
                                       cfg.SEED_MODELO, guardar_modelo=True)
    print(f"Modelo final registrado en la run {final['run_id']}")
    vista.resumen_clusters(final["predicciones"], centros(final["modelo"]))

    vista.comentar(
        "La métrica nueva es el índice Davies-Bouldin: para cada cluster busca el vecino con el que "
        "más se 'pisa' ((dispersión_i + dispersión_j) / distancia entre centros) y hace la media. "
        "Cuanto más bajo, mejor, así que se devuelve tal cual como loss (Hyperopt minimiza). "
        "Cada intento es una run del experimento kmeans_BBVA. Hay dos semillas: la del modelo fija "
        "los centroides iniciales de KMeans y la de Hyperopt fija qué k se prueban.",
        "Enunciado-Practica4.pdf (Hyperopt, Reproducibilidad: la segunda semilla); "
        "teoriaCOMPLETO.pdf (dispersión intracluster e intercluster)")
    return mejor_k, final


def ej3():
    vista.cabecera("Ej3", "Recuperar las runs de un experimento ordenadas por la métrica")
    '''
    Crea una función que permita recuperar todas las runs ejecutadas en un proyecto, ordénalas por
    el valor de la métrica, e imprime por pantalla el id de cada run, seguido de los parámetros
    que has almacenado y el valor de la métrica obtenida.
    '''
    print(f"--- Experimento {cfg.EXPERIMENTO_EJ2} (Davies-Bouldin, de mejor a peor) ---")
    runs = obtener_runs_ordenadas(cfg.EXPERIMENTO_EJ2, "davies_bouldin", ascendente=True)
    vista.runs_ordenadas(runs, "davies_bouldin")

    print(f"\n--- Experimento {cfg.EXPERIMENTO_EJ1} (silhouette, de mejor a peor) ---")
    runs = obtener_runs_ordenadas(cfg.EXPERIMENTO_EJ1, "silhouette", ascendente=False)
    vista.runs_ordenadas(runs, "silhouette")


def ej4(spark, df_bbva, mejor_k, final_ej2):
    vista.cabecera("Ej4 (opcional)", "Feature store con versiones y reproducibilidad")
    '''
    Crea funciones para guardar las features usadas en los ej anteriores con una columna version,
    y para recuperar una versión concreta. Guarda la versión 1, recupérala, reentrena y comprueba
    (gracias a la seed fija) que obtienes el mismo resultado. Explica en un comentario por qué
    esto favorece la reproducibilidad.
    '''
    configurar_mlflow(cfg.MLRUNS_DIR, cfg.EXPERIMENTO_EJ2)

    guardar_features(df_bbva, 1, cfg.RUTA_FEATURES)
    print("Versiones en el feature store:", versiones_disponibles(spark, cfg.RUTA_FEATURES))

    features_v1 = leer_features(spark, cfg.RUTA_FEATURES, 1, cfg.TICKER_EJ1)
    print("Filas recuperadas de la versión 1:", features_v1.count())

    reentreno = entrenar_evaluar_registrar(features_v1, cfg.TICKER_EJ1, cfg.COLUMNAS_EJ1,
                                           mejor_k, cfg.SEED_MODELO, origen="feature_store_v1")

    # Comparación día a día: ¿cae cada fecha en el mismo cluster en los dos modelos?
    original = final_ej2["predicciones"].select("Fecha", final_ej2["predicciones"]["prediction"].alias("p1"))
    nuevo = reentreno["predicciones"].select("Fecha", reentreno["predicciones"]["prediction"].alias("p2"))
    distintos = original.join(nuevo, "Fecha").filter("p1 <> p2").count()

    print(f"Davies-Bouldin original: {final_ej2['davies_bouldin']:.6f} | "
          f"reentrenado: {reentreno['davies_bouldin']:.6f}")
    print(f"Silhouette     original: {final_ej2['silhouette']:.6f} | "
          f"reentrenado: {reentreno['silhouette']:.6f}")
    print("Días con cluster distinto:", distintos)

    vista.comentar(
        "Las features se guardan una vez con su clave (Ticker, Fecha) y un número de versión. Para "
        "reentrenar se lee la versión 1 en lugar de recalcularla, así que el modelo usa exactamente "
        "los mismos datos aunque mañana cambie el código que los prepara. Junto con la semilla fija "
        "de KMeans, mismos datos + mismo código + misma semilla dan el mismo modelo, y la run guarda "
        "origen_datos=feature_store_v1 para saber de dónde salió.",
        "Enunciado-Practica4.pdf (Feature Store, Reproducibilidad)")


def ej5(df_gold):
    vista.cabecera("Ej5 (opcional)", "Mismo algoritmo sobre otra empresa en otro experimento")
    '''
    Repite el entrenamiento con MLflow + Hyperopt usando otro conjunto de datos (otra empresa de la
    gold u otras columnas) y guárdalo en un experimento distinto. Muestra la lista de experimentos
    disponibles y, para cada uno, su mejor run. Comenta cómo la separación en experimentos permite
    comparar el mismo algoritmo sobre datos distintos.
    '''
    configurar_mlflow(cfg.MLRUNS_DIR, cfg.EXPERIMENTO_EJ5)

    df_san = datos_empresa(df_gold, cfg.TICKER_EJ5, cfg.COLUMNAS_EJ5)
    print(f"Filas {cfg.TICKER_EJ5}:", df_san.count())

    mejor_k, _ = buscar_mejor_k(df_san, cfg.TICKER_EJ5, cfg.COLUMNAS_EJ5,
                                cfg.K_MIN, cfg.K_MAX, cfg.MAX_EVALS,
                                cfg.SEED_MODELO, cfg.SEED_HYPEROPT)
    print(f"\nk seleccionado por Hyperopt para {cfg.TICKER_EJ5}: {mejor_k}")
    final = entrenar_evaluar_registrar(df_san, cfg.TICKER_EJ5, cfg.COLUMNAS_EJ5, mejor_k,
                                       cfg.SEED_MODELO, guardar_modelo=True)

    print("\nExperimentos disponibles y su mejor run (según Davies-Bouldin):")
    lista = [(exp, mejor_run(exp.name, "davies_bouldin", ascendente=True), "davies_bouldin")
             for exp in listar_experimentos()]
    vista.experimentos_y_mejores(lista)

    vista.comentar(
        "Cada experimento agrupa las runs de un mismo problema (algoritmo + dato). Al separar "
        "kmeans_BBVA y kmeans_SAN, las runs de una empresa no se mezclan con las de la otra: se "
        "puede pedir la mejor run de cada experimento y comparar, con la misma métrica y el mismo "
        "código, qué k sale para cada empresa y con qué calidad.",
        "Enunciado-Practica4.pdf (MLflow: experimentos y runs)")
    return final


def extra_exportar(predicciones):
    vista.cabecera("Extra (Clase 3)", "Exportar los clusters a Parquet para Power BI")
    df = exportar_clusters(predicciones, cfg.RUTA_EXPORT_CLUSTERS)
    print("Filas exportadas:", df.count(), "->", cfg.RUTA_EXPORT_CLUSTERS)


def main():
    spark = crear_spark_session()

    df_gold = cargar_gold(spark, cfg.RUTA_FACT)
    df_bbva = datos_empresa(df_gold, cfg.TICKER_EJ1, cfg.COLUMNAS_EJ1).cache()

    ej1(df_bbva)
    mejor_k, final_ej2 = ej2(df_bbva)
    ej3()
    ej4(spark, df_bbva, mejor_k, final_ej2)
    final_ej5 = ej5(df_gold)
    extra_exportar([final_ej2["predicciones"], final_ej5["predicciones"]])

    spark.stop()


if __name__ == "__main__":
    main()
