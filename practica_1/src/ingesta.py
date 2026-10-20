import time
from datetime import datetime
from functools import reduce

import pandas as pd
import yfinance as yf
from pyspark.sql import DataFrame
from pyspark.sql.types import (StructType, StructField, DateType, DoubleType,
                               LongType, StringType)

#EJ1-a 
ESQUEMA_PRECIOS = StructType([
    StructField("Fecha", DateType(), False),
    StructField("Open", DoubleType(), True),
    StructField("High", DoubleType(), True),
    StructField("Low", DoubleType(), True),
    StructField("Close", DoubleType(), True),
    StructField("Volume", LongType(), True),
    StructField("Ticker", StringType(), False),
    StructField("Dividends", DoubleType(), False),
    StructField("Stock_Splits", DoubleType(), False),
])



def _numero(valor):
    return None if pd.isna(valor) else float(valor)

#EJ 1A 
def descargar_historico(spark, ticker, inicio, fin):
    fin_incluido = (datetime.strptime(fin, "%Y-%m-%d") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    '''
    DECLARACIÓN USO DE IA
    Pedido a claude, ya que me fallaba en el primer intento de busqueda
    y de esta manera tengo hasta tres intentos
    '''

    pdf = pd.DataFrame()
    for intento in range(3):
        try:
            pdf = yf.Ticker(ticker).history(start = inicio, end = fin_incluido, interval = "1d", auto_adjust = False)

        except Exception:
            pdf = pd.DataFrame()
        if not pdf.empty:
            break
        time.sleep(2* (intento + 1))

    #si no hay datos devolemeos tabla vacia
    if pdf.empty:
        return spark.createDataFrame([], ESQUEMA_PRECIOS)

    pdf= pdf.reset_index()
    pdf = pdf.rename(columns={pdf.columns[0]: "Fecha", "Stock Splits": "Stock_Splits" })
    #quita hora y zoanas horarias, dejando solamente la fecha
    pdf["Fecha"] = pd.to_datetime(pdf["Fecha"]).dt.date       

    filas = []
    for r in pdf.itertuples(index = False):
        filas.append((
            r.Fecha, _numero(r.Open), _numero(r.High), _numero(r.Low), _numero(r.Close),
            None if pd.isna(r.Volume) else int(r.Volume),
            ticker,
            0.0 if pd.isna(r.Dividends) else float(r.Dividends),
            0.0 if pd.isna(r.Stock_Splits) else float(r.Stock_Splits),
        ))

    return spark.createDataFrame(filas, ESQUEMA_PRECIOS)

#EJ 1B
def descargar_ibex(spark, tickers, inicio, fin):
    dfs = [descargar_historico(spark, t, inicio, fin) for t in tickers]
    return reduce(DataFrame.unionByName, dfs)

