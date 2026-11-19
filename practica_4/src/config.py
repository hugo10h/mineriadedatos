"""
config.py  ->  Configuración central de la Práctica 4 (MLOps con Spark).

Aquí solo hay CONSTANTES: rutas, semillas, columnas y nombres de experimentos.
Si mañana cambias de empresa, de columnas o de rango de k, se toca este fichero
y el resto del código no cambia (responsabilidad única).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
# practica_4/  (config.py está en practica_4/src, así que subimos dos niveles)
PROYECTO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROYECTO_DIR / "data"

# Entrada: la capa gold de la Práctica 2, copiada en Parquet dentro de esta práctica
# (practica_4/data/gold/fact_cotizaciones). Así el proyecto se entrega autocontenido
# y la parte de ML queda DESPUÉS de la gold, sin tocar el pipeline de datos de la P1/P2.
RUTA_FACT = (DATA_DIR / "gold" / "fact_cotizaciones").as_posix()

# Feature store "hecho a mano" (Ej4): tabla Delta con clave (Ticker, Fecha) + columna version
RUTA_FEATURES = (DATA_DIR / "lake" / "features").as_posix()

# Salida para Power BI (Clase 3): cada día con el cluster que le ha tocado
RUTA_EXPORT_CLUSTERS = (DATA_DIR / "export" / "clusters").as_posix()

# Carpeta donde MLflow guarda experimentos, runs, métricas y modelos
MLRUNS_DIR = PROYECTO_DIR / "mlruns"

# ---------------------------------------------------------------------------
# SEMILLAS (reproducibilidad)
# ---------------------------------------------------------------------------
SEED_GLOBAL = 42      # random de Python y numpy
SEED_MODELO = 1       
SEED_HYPEROPT = 42    

# ---------------------------------------------------------------------------
# DATOS DEL MODELO
# ---------------------------------------------------------------------------
TICKER_EJ1 = "BBVA.MC"
COLUMNAS_EJ1 = ["Open", "Close"]       

TICKER_EJ5 = "SAN.MC"                  
COLUMNAS_EJ5 = ["Open", "Close"]

# ---------------------------------------------------------------------------
# BÚSQUEDA DEL NÚMERO DE CLUSTERS
# ---------------------------------------------------------------------------
K_CANDIDATOS_EJ1 = list(range(2, 9))  
K_MIN, K_MAX = 2, 10                   
MAX_EVALS = 12                         

# ---------------------------------------------------------------------------
# MLFLOW: un experimento por algoritmo + dato (+ forma de buscar)
# ---------------------------------------------------------------------------
EXPERIMENTO_EJ1 = "kmeans_BBVA_manual"     
EXPERIMENTO_EJ2 = "kmeans_BBVA"            
EXPERIMENTO_EJ5 = "kmeans_SAN"             
