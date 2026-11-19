"""
vista.py  ->  VISTA: todo lo que se imprime por pantalla. Aquí no se calcula nada.
"""


def cabecera(ejercicio, titulo):
    print("\n" + "=" * 90)
    print(f"{ejercicio}: {titulo}")
    print("=" * 90)


def comentar(texto, referencia):
    # Formato de comentario que pide la normativa
    print("Comentario/Reflexion/Investigacion")
    print(texto)
    print("Referencia/s")
    print(referencia)


def tabla_k(resultados):
    """Ej1: una fila por k con sus métricas."""
    print(f"{'k':>3} | {'silhouette':>10} | {'davies_bouldin':>14} | {'sse':>10}")
    print("-" * 48)
    for r in resultados:
        print(f"{r['k']:>3} | {r['silhouette']:>10.4f} | {r['davies_bouldin']:>14.4f} | {r['sse']:>10.2f}")


def resumen_clusters(predicciones, centros):
    """Cuántos días caen en cada cluster y dónde está su centro (Open, Close)."""
    print("Días por cluster:")
    predicciones.groupBy("prediction").count().orderBy("prediction").show()
    for i, c in enumerate(centros):
        print(f"  Cluster {i}: centro -> " + ", ".join(f"{v:.3f}" for v in c))


def runs_ordenadas(runs, metrica):
    """Ej3: id de cada run + parámetros almacenados + valor de la métrica."""
    columnas_param = sorted(c for c in runs.columns if c.startswith("params."))
    for _, fila in runs.iterrows():
        params = ", ".join(f"{c.replace('params.', '')}={fila[c]}" for c in columnas_param)
        print(f"Run ID: {fila['run_id']} | {params} | {metrica}: {fila['metrics.' + metrica]:.4f}")


def experimentos_y_mejores(lista):
    """Ej5: cada experimento con su mejor run."""
    for exp, mejor, metrica in lista:
        print(f"Experimento '{exp.name}' (id={exp.experiment_id})")
        if mejor is None:
            print("   (sin runs con esa métrica)")
            continue
        print(f"   Mejor run: {mejor['run_id']} | ticker={mejor['params.ticker']} "
              f"| k={mejor['params.k']} | {metrica}={mejor['metrics.' + metrica]:.4f} "
              f"| silhouette={mejor['metrics.silhouette']:.4f}")
