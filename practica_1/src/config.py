"""Configuración central del proyecto: rutas del data lake, tickers y base de datos."""
import os
from pathlib import Path

PROYECTO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROYECTO_DIR / "data"
LAKE_DIR = DATA_DIR / "lake"


def ruta_lake(capa, tabla):
    return (LAKE_DIR / capa / tabla).as_posix()


BRONZE_IBEX = ruta_lake("bronze", "ibex")
BRONZE_SENTIMIENTO = ruta_lake("bronze", "sentimiento")
#SILVER_IBEX = ruta_lake("silver", "ibex")
#SILVER_SENTIMIENTO = ruta_lake("silver", "sentimiento")

CSV_SENTIMIENTO = (DATA_DIR / "sentimiento_ibex.csv").as_posix()

#Para el ejercicio 1b
FECHA_FIN = "2026-10-01"

#DECLARACIÓN USO DE IA--> Pedi a claude que me hiciese el listado.
TICKERS_IBEX = [
    "IBE.MC", "REP.MC", "NTGY.MC", "ELE.MC", "ENG.MC", "RED.MC",      # energía / utilities
    "SAN.MC", "BBVA.MC", "CABK.MC", "BKT.MC", "SAB.MC", "UNI.MC", "MAP.MC",  # finanzas
    "ACS.MC", "ANA.MC", "ANE.MC", "ACX.MC", "MTS.MC", "SCYR.MC",      # construcción / industria
    "CLNX.MC", "TEF.MC",                                              # telecomunicaciones
    "AENA.MC", "FER.MC", "ITX.MC", "AMS.MC", "IAG.MC", "MEL.MC",      # turismo / consumo
    "GRF.MC", "FDR.MC", "SLR.MC", "ROVI.MC", "LOG.MC", "IDR.MC", "PUIG.MC",
    "COL.MC", "MRL.MC",                                               # inmobiliario
]

# MySQL (capa gold -> JDBC). Se pueden sobreescribir con variables de entorno.
#DB_HOST = os.environ.get("DB_HOST", "localhost")
#DB_PORT = int(os.environ.get("DB_PORT", "3306"))
#DB_NAME = os.environ.get("DB_NAME", "ibex35")
#DB_USER = os.environ.get("DB_USER", "spark_user")
#DB_PASSWORD = os.environ.get("DB_PASSWORD", "Ibex35Pass1!")
